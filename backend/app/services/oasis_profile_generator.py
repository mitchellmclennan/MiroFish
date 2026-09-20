"""
OASIS Agent Profile生成器
将Zep图谱中的实体转换为OASIS模拟平台所需的Agent Profile格式

优化改进：
1. 调用Zep检索功能二次丰富节点信息
2. 优化提示词生成非常详细的人设
3. 区分个人实体和抽象群体实体
4. 图谱事实注入提示词 + 明确的禁止虚构规则 + 可审计的溯源记录
"""

import json
import random
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI
from ..config import Config
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction, get_locale, is_english_forced, set_locale, t
from ..utils.openai_chat_compat import create_chat_completion, extract_chat_completion_text
from ..utils.zep import (
    call_zep_read_with_retry,
    get_zep_client,
    is_retryable_zep_error,
    normalize_zep_search_query,
)
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('mirofish.oasis_profile')

# Retrieval query used when the run is forced to English. Mirrors
# locales/en.json progress.zepSearchQuery so retrieval does not depend on
# the requesting thread's UI locale (a Chinese-language query would not
# match an English-language graph).
ENGLISH_SEARCH_QUERY_TEMPLATE = (
    "All information, activities, events, relationships and background about {name}"
)

# 显式禁止虚构规则：事实性内容必须来自图谱上下文，缺失的信息必须泛化描述。
ENGLISH_GROUNDING_RULES = """## STRICT GROUNDING RULES (no invention)
- Every claim about this entity's identity, role, history, relationships, and involvement in the event MUST be supported by the graph context (source facts) above.
- Do NOT invent events, relationships, dates, prices, product details, or personal history that are not present in that context.
- If a specific detail is missing from the context, describe it generically instead of fabricating specifics.
- Personality traits, posting habits, and style may be fleshed out ONLY where the context does not contradict them."""


@dataclass
class PersonaProvenance:
    """Auditable record of which graph facts grounded a persona.

    Written per persona to the log (``PERSONA_PROVENANCE`` lines) and to the
    run's ``persona_provenance.json`` sidecar so spot-checks can trace every
    persona claim back to knowledge-graph facts.
    """
    entity_uuid: str
    entity_name: str
    entity_type: Optional[str] = None
    llm_used: bool = False
    model: Optional[str] = None
    search_attempted: bool = False
    search_query: Optional[str] = None
    search_facts_returned: int = 0
    related_edge_count: int = 0
    facts: List[Dict[str, Any]] = field(default_factory=list)
    context_chars: int = 0
    context_truncated: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_uuid": self.entity_uuid,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "llm_used": self.llm_used,
            "model": self.model,
            "search_attempted": self.search_attempted,
            "search_query": self.search_query,
            "search_facts_returned": self.search_facts_returned,
            "related_edge_count": self.related_edge_count,
            "facts": self.facts,
            "context_chars": self.context_chars,
            "context_truncated": self.context_truncated,
            "error": self.error,
        }


@dataclass
class EntityGroundingContext:
    """Entity context assembled for the persona prompt plus its fact ledger."""
    context_text: str
    facts: List[Dict[str, Any]] = field(default_factory=list)
    search_attempted: bool = False
    search_query: Optional[str] = None
    search_facts_returned: int = 0


def _coerce_to_str(value: Any) -> str:
    """Coerce a value to a plain string.

    Handles dict, list, and other non-string types that may be returned
    by LLM JSON parsing.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ('text', 'value', 'description', 'content', 'summary', 'name'):
            if key in value:
                candidate = _coerce_to_str(value[key])
                if candidate:
                    return candidate
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        str_items = [_coerce_to_str(item) for item in value]
        str_items = [item for item in str_items if item]
        return ', '.join(str_items)
    return str(value)


def _coerce_to_str_list(value: Any) -> List[str]:
    """Coerce a value to a list of strings.

    Handles nested structures that may be returned by LLM JSON parsing.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        result: List[str] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                result.extend(_coerce_to_str_list(item))
            else:
                text = _coerce_to_str(item)
                if text:
                    result.append(text)
        return result
    text = _coerce_to_str(value)
    return [text] if text else []


@dataclass
class OasisAgentProfile:
    """OASIS Agent Profile数据结构"""
    # 通用字段
    user_id: int
    user_name: str
    name: str
    bio: str
    persona: str

    # 可选字段 - Reddit风格
    karma: int = 1000
    
    # 可选字段 - Twitter风格
    friend_count: int = 100
    follower_count: int = 150
    statuses_count: int = 500
    
    # 额外人设信息
    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None
    country: Optional[str] = None
    profession: Optional[str] = None
    interested_topics: List[str] = field(default_factory=list)
    
    # 来源实体信息
    source_entity_uuid: Optional[str] = None
    source_entity_type: Optional[str] = None

    # 溯源信息：生成该人设时注入提示词的图谱事实（不进入OASIS输出格式）
    provenance: Dict[str, Any] = field(default_factory=dict)

    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    
    def __post_init__(self):
        """Normalize structured LLM fields once at the profile boundary."""
        self.bio = _coerce_to_str(self.bio) or self.name
        self.persona = _coerce_to_str(self.persona) or (
            f"{self.name} is a participant in social discussions."
        )
        self.country = _coerce_to_str(self.country) or None
        self.profession = _coerce_to_str(self.profession) or None
        self.gender = _coerce_to_str(self.gender) or None
        self.mbti = _coerce_to_str(self.mbti) or None
        self.interested_topics = _coerce_to_str_list(self.interested_topics)

    def to_reddit_format(self) -> Dict[str, Any]:
        """转换为Reddit平台格式"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # OASIS 库要求字段名为 username（无下划线）
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "created_at": self.created_at,
        }
        
        # 添加额外人设信息（如果有）
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics
        
        return profile
    
    def to_twitter_format(self) -> Dict[str, Any]:
        """转换为Twitter平台格式"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # OASIS 库要求字段名为 username（无下划线）
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "created_at": self.created_at,
        }
        
        # 添加额外人设信息
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics
        
        return profile
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为完整字典格式"""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "age": self.age,
            "gender": self.gender,
            "mbti": self.mbti,
            "country": self.country,
            "profession": self.profession,
            "interested_topics": self.interested_topics,
            "source_entity_uuid": self.source_entity_uuid,
            "source_entity_type": self.source_entity_type,
            "provenance": self.provenance,
            "created_at": self.created_at,
        }


class OasisProfileGenerator:
    """
    OASIS Profile生成器
    
    将Zep图谱中的实体转换为OASIS模拟所需的Agent Profile
    
    优化特性：
    1. 调用Zep图谱检索功能获取更丰富的上下文
    2. 生成非常详细的人设（包括基本信息、职业经历、性格特征、社交媒体行为等）
    3. 区分个人实体和抽象群体实体
    4. 图谱事实注入提示词，明确禁止虚构，并记录可审计的溯源信息
    """
    
    # 注入人设提示词的上下文最大长度（与提示词构建中的截断保持一致）
    CONTEXT_PROMPT_CHAR_LIMIT = 3000
    
    # MBTI类型列表
    MBTI_TYPES = [
        "INTJ", "INTP", "ENTJ", "ENTP",
        "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",
        "ISTP", "ISFP", "ESTP", "ESFP"
    ]
    
    # 常见国家列表
    COUNTRIES = [
        "China", "US", "UK", "Japan", "Germany", "France", 
        "Canada", "Australia", "Brazil", "India", "South Korea"
    ]
    
    # 个人类型实体（需要生成具体人设）
    INDIVIDUAL_ENTITY_TYPES = [
        "student", "alumni", "professor", "person", "publicfigure", 
        "expert", "faculty", "official", "journalist", "activist"
    ]
    
    # 群体/机构类型实体（需要生成群体代表人设）
    GROUP_ENTITY_TYPES = [
        "university", "governmentagency", "organization", "ngo", 
        "mediaoutlet", "company", "institution", "group", "community"
    ]
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        zep_api_key: Optional[str] = None,
        graph_id: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Zep客户端用于检索丰富上下文
        self.zep_api_key = zep_api_key or Config.ZEP_API_KEY
        self.zep_client = None
        self.graph_id = graph_id
        
        if self.zep_api_key:
            try:
                self.zep_client = get_zep_client(self.zep_api_key)
            except Exception as e:
                logger.warning(f"Zep客户端初始化失败: {e}")
    
    def generate_profile_from_entity(
        self, 
        entity: EntityNode, 
        user_id: int,
        use_llm: bool = True
    ) -> OasisAgentProfile:
        """
        从Zep实体生成OASIS Agent Profile
        
        Args:
            entity: Zep实体节点
            user_id: 用户ID（用于OASIS）
            use_llm: 是否使用LLM生成详细人设
            
        Returns:
            OasisAgentProfile
        """
        entity_type = entity.get_entity_type() or "Entity"
        
        # 基础信息
        name = entity.name
        user_name = self._generate_username(name)
        
        # 构建上下文信息（同时生成事实台账用于溯源）
        grounding = self._build_grounding_context(entity)
        context = grounding.context_text

        profile_data: Dict[str, Any]
        llm_used = False
        if use_llm:
            # 使用LLM生成详细人设（内部失败时自动回退规则生成并打标记）
            profile_data = self._generate_profile_with_llm(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes,
                context=context
            )
            llm_used = not profile_data.pop("_rule_based_fallback", False)
        else:
            # 使用规则生成基础人设
            profile_data = self._generate_profile_rule_based(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes
            )

        # 构建溯源记录：本条人设的每个事实依据都可回查
        provenance = PersonaProvenance(
            entity_uuid=entity.uuid,
            entity_name=name,
            entity_type=entity_type,
            llm_used=llm_used,
            model=self.model_name if llm_used else None,
            search_attempted=grounding.search_attempted,
            search_query=grounding.search_query,
            search_facts_returned=grounding.search_facts_returned,
            related_edge_count=len(entity.related_edges or []),
            facts=grounding.facts,
            context_chars=len(context),
            context_truncated=len(context) > self.CONTEXT_PROMPT_CHAR_LIMIT,
            error=None,
        )
        logger.info(
            "PERSONA_PROVENANCE " + json.dumps(provenance.to_dict(), ensure_ascii=False)
        )

        return OasisAgentProfile(
            user_id=user_id,
            user_name=user_name,
            name=name,
            bio=profile_data.get("bio", f"{entity_type}: {name}"),
            persona=profile_data.get("persona", entity.summary or f"A {entity_type} named {name}."),
            karma=profile_data.get("karma", random.randint(500, 5000)),
            friend_count=profile_data.get("friend_count", random.randint(50, 500)),
            follower_count=profile_data.get("follower_count", random.randint(100, 1000)),
            statuses_count=profile_data.get("statuses_count", random.randint(100, 2000)),
            age=profile_data.get("age"),
            gender=profile_data.get("gender"),
            mbti=profile_data.get("mbti"),
            country=profile_data.get("country"),
            profession=profile_data.get("profession"),
            interested_topics=profile_data.get("interested_topics", []),
            source_entity_uuid=entity.uuid,
            source_entity_type=entity_type,
            provenance=provenance.to_dict(),
        )
    
    def _generate_username(self, name: str) -> str:
        """生成用户名"""
        # 移除特殊字符，转换为小写
        username = name.lower().replace(" ", "_")
        username = ''.join(c for c in username if c.isalnum() or c == '_')
        
        # 添加随机后缀避免重复
        suffix = random.randint(100, 999)
        return f"{username}_{suffix}"
    
    @staticmethod
    def _extract_search_fact(item: Any) -> Optional[str]:
        """Extract a fact string from one search result item.

        Handles both Zep Cloud SDK objects (``.fact``) and OpenZep local
        compatibility payloads, which return plain dicts under a ``results``
        key instead of ``edges``/``nodes``.
        """
        if isinstance(item, dict):
            fact = item.get("fact")
            return fact if isinstance(fact, str) and fact.strip() else None
        fact = getattr(item, "fact", None)
        return fact if isinstance(fact, str) and fact.strip() else None

    @staticmethod
    def _extract_search_node_summaries(
        result: Any, entity_name: str
    ) -> List[str]:
        """Extract node summaries from a node-scope search result.

        Zep Cloud returns node objects with ``name``/``summary``. OpenZep
        local ignores ``scope`` and returns edge-fact dicts in ``results``,
        so only real node payloads contribute summaries here.
        """
        summaries: List[str] = []
        seen = {entity_name}
        items = None
        for attr in ("nodes", "results"):
            items = getattr(result, attr, None)
            if items:
                break
        if not items:
            return summaries
        for item in items:
            if isinstance(item, dict):
                name = item.get("name")
                summary = item.get("summary")
            else:
                name = getattr(item, "name", None)
                summary = getattr(item, "summary", None)
            if not isinstance(summary, str) or not summary.strip():
                continue
            if isinstance(name, str) and name and name not in seen:
                summaries.append(f"{name}: {summary}")
                seen.add(name)
            elif summary not in seen:
                summaries.append(summary)
                seen.add(summary)
        return summaries

    def _search_zep_for_entity(self, entity: EntityNode) -> Dict[str, Any]:
        """
        使用Zep图谱混合搜索功能获取实体相关的丰富信息

        Zep没有内置混合搜索接口，需要分别搜索edges和nodes然后合并结果。
        使用并行请求同时搜索，提高效率。

        Args:
            entity: 实体节点对象

        Returns:
            包含facts, node_summaries, context, query的字典
        """
        import concurrent.futures

        empty = {
            "facts": [],
            "node_summaries": [],
            "context": "",
            "query": None,
            "attempted": False,
        }

        if not self.zep_client:
            return empty

        entity_name = entity.name

        results = {
            "facts": [],
            "node_summaries": [],
            "context": "",
            "query": None,
            "attempted": True,
        }

        # 必须有graph_id才能进行搜索
        if not self.graph_id:
            logger.debug(f"跳过Zep检索：未设置graph_id")
            results["attempted"] = False
            return results

        if is_english_forced():
            # Retrieval must not depend on UI locale: a Chinese template
            # query cannot match an English-language graph.
            comprehensive_query = normalize_zep_search_query(
                ENGLISH_SEARCH_QUERY_TEMPLATE.format(name=entity_name)
            )
        else:
            comprehensive_query = normalize_zep_search_query(
                t('progress.zepSearchQuery', name=entity_name)
            )
        results["query"] = comprehensive_query

        def search_edges():
            """搜索边（事实/关系）- 带重试机制"""
            return call_zep_read_with_retry(
                lambda: self.zep_client.graph.search(
                        query=comprehensive_query,
                        graph_id=self.graph_id,
                        limit=30,
                        scope="edges",
                        reranker="rrf"
                ),
                operation_name=f"profile edge search ({entity.uuid})",
            )

        def search_nodes():
            """搜索节点（实体摘要）- 带重试机制"""
            return call_zep_read_with_retry(
                lambda: self.zep_client.graph.search(
                        query=comprehensive_query,
                        graph_id=self.graph_id,
                        limit=20,
                        scope="nodes",
                        reranker="rrf"
                ),
                operation_name=f"profile node search ({entity.uuid})",
            )

        try:
            # 并行执行edges和nodes搜索
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                edge_future = executor.submit(search_edges)
                node_future = executor.submit(search_nodes)

                # 获取结果
                # Each request already has the configured HTTP timeout and
                # typed retry budget. A second hard-coded 30s future timeout
                # discarded late successes while the executor still waited.
                edge_result = edge_future.result()
                node_result = node_future.result()

            # 处理边搜索结果。
            # Zep Cloud在.edges上返回SDK对象；OpenZep local忽略scope并在
            # .results上返回原始dict——两种形状都要读取，否则本地环境的
            # 检索结果会被静默丢弃。
            all_facts = []
            seen_facts = set()
            for result, source_attr in ((edge_result, "edges"), (edge_result, "results")):
                items = getattr(result, source_attr, None) if result is not None else None
                if not items:
                    continue
                for item in items:
                    fact = self._extract_search_fact(item)
                    if fact and fact not in seen_facts:
                        seen_facts.add(fact)
                        all_facts.append(fact)
            results["facts"] = all_facts

            # 处理节点搜索结果（仅真实的节点负载会贡献摘要）
            node_summaries = self._extract_search_node_summaries(node_result, entity_name)
            if not node_summaries:
                # OpenZep local在节点scope也返回事实dict，退化为事实处理
                for item in (getattr(node_result, "results", None) or []):
                    fact = self._extract_search_fact(item)
                    if fact and fact not in seen_facts:
                        seen_facts.add(fact)
                        results["facts"].append(fact)
            results["node_summaries"] = node_summaries

            # 构建综合上下文
            context_parts = []
            if results["facts"]:
                header = (
                    "Facts retrieved from the knowledge graph (Zep search):"
                    if is_english_forced()
                    else "Zep检索到的事实信息"
                )
                context_parts.append(header + "\n" + "\n".join(f"- {f}" for f in results["facts"][:20]))
            if results["node_summaries"]:
                header = (
                    "Related nodes retrieved from the knowledge graph (Zep search):"
                    if is_english_forced()
                    else "Zep检索到的相关节点"
                )
                context_parts.append(header + "\n" + "\n".join(f"- {s}" for s in results["node_summaries"][:10]))
            results["context"] = "\n\n".join(context_parts)

            logger.info(f"Zep混合检索完成: {entity_name}, 获取 {len(results['facts'])} 条事实, {len(results['node_summaries'])} 个相关节点")

        except Exception as e:
            logger.warning(f"Zep检索失败 ({entity_name}): {e}")
            if not is_retryable_zep_error(e):
                raise

        return results
    
    def _build_grounding_context(self, entity: EntityNode) -> EntityGroundingContext:
        """
        构建实体的完整上下文信息并记录事实台账（用于溯源）

        包括：
        1. 实体本身的边信息（事实）
        2. Zep混合检索到的相关事实
        3. 关联节点的详细信息
        4. 实体属性

        事实台账中的每条记录都会进入溯源文件，便于人工核查
        人设中的每个事实性描述是否有图谱依据。
        """
        context_parts = []
        facts: List[Dict[str, Any]] = []
        english = is_english_forced()

        # 1. 添加实体属性信息
        if entity.attributes:
            attrs = []
            for key, value in entity.attributes.items():
                if value and str(value).strip():
                    attrs.append(f"- {key}: {value}")
            if attrs:
                header = "### Entity attributes" if english else "### 实体属性"
                context_parts.append(header + "\n" + "\n".join(attrs))

        # 2. 添加相关边信息（事实/关系）
        existing_facts = set()
        if entity.related_edges:
            relationships = []
            for edge in entity.related_edges:  # 不限制数量
                fact = edge.get("fact", "")
                edge_name = edge.get("edge_name", "")
                direction = edge.get("direction", "")

                if fact:
                    relationships.append(f"- {fact}")
                    existing_facts.add(fact)
                elif edge_name:
                    if direction == "outgoing":
                        relationships.append(f"- {entity.name} --[{edge_name}]--> (相关实体)")
                    else:
                        relationships.append(f"- (相关实体) --[{edge_name}]--> {entity.name}")

            if relationships:
                header = (
                    "### Facts and relationships from the knowledge graph"
                    if english
                    else "### 相关事实和关系"
                )
                context_parts.append(header + "\n" + "\n".join(relationships))

        # 3. 使用Zep混合检索获取更丰富的信息
        zep_results = self._search_zep_for_entity(entity)
        # 去重：排除已存在的边事实，只注入新事实
        new_facts = [f for f in zep_results.get("facts", []) if f not in existing_facts]
        zep_summaries = list(zep_results.get("node_summaries", []))
        if new_facts or zep_summaries:
            retrieval_blocks = []
            if new_facts:
                header = (
                    "Facts retrieved from the knowledge graph (Zep search):"
                    if english
                    else "Zep检索到的事实信息"
                )
                retrieval_blocks.append(header + "\n" + "\n".join(f"- {f}" for f in new_facts[:15]))
            if zep_summaries:
                header = (
                    "Related nodes retrieved from the knowledge graph (Zep search):"
                    if english
                    else "Zep检索到的相关节点"
                )
                retrieval_blocks.append(header + "\n" + "\n".join(f"- {s}" for s in zep_summaries[:10]))
            context_parts.append("\n\n".join(retrieval_blocks))

        # 4. 添加关联节点的详细信息（放在检索事实之后，截断时优先损失低价值信息）
        if entity.related_nodes:
            related_info = []
            for node in entity.related_nodes:  # 不限制数量
                node_name = node.get("name", "")
                node_labels = node.get("labels", [])
                node_summary = node.get("summary", "")

                # 过滤掉默认标签
                custom_labels = [l for l in node_labels if l not in ["Entity", "Node"]]
                label_str = f" ({', '.join(custom_labels)})" if custom_labels else ""

                if node_summary:
                    related_info.append(f"- **{node_name}**{label_str}: {node_summary}")
                else:
                    related_info.append(f"- **{node_name}**{label_str}")

            if related_info:
                header = "### Related entities" if english else "### 关联实体信息"
                context_parts.append(header + "\n" + "\n".join(related_info))

        context_text = "\n\n".join(part for part in context_parts if part)

        # 构建事实台账（溯源依据）
        for fact in existing_facts:
            facts.append({"text": fact, "source": "related_edge"})
        for fact in new_facts:
            facts.append({"text": fact, "source": "zep_search"})
        for summary in zep_summaries:
            facts.append({"text": summary, "source": "zep_search_node_summary"})
        if entity.attributes:
            for key, value in entity.attributes.items():
                if value and str(value).strip():
                    facts.append({"text": f"{key}: {value}", "source": "entity_attribute"})

        return EntityGroundingContext(
            context_text=context_text,
            facts=facts,
            search_attempted=bool(zep_results.get("attempted")),
            search_query=zep_results.get("query"),
            search_facts_returned=len(zep_results.get("facts", [])),
        )

    def _build_entity_context(self, entity: EntityNode) -> str:
        """构建实体的完整上下文信息（兼容包装，返回纯文本上下文）"""
        return self._build_grounding_context(entity).context_text
    
    def _is_individual_entity(self, entity_type: str) -> bool:
        """判断是否是个人类型实体"""
        return entity_type.lower() in self.INDIVIDUAL_ENTITY_TYPES
    
    def _is_group_entity(self, entity_type: str) -> bool:
        """判断是否是群体/机构类型实体"""
        return entity_type.lower() in self.GROUP_ENTITY_TYPES
    
    def _generate_profile_with_llm(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> Dict[str, Any]:
        """
        使用LLM生成非常详细的人设
        
        根据实体类型区分：
        - 个人实体：生成具体的人物设定
        - 群体/机构实体：生成代表性账号设定
        """
        
        is_individual = self._is_individual_entity(entity_type)
        
        if is_individual:
            prompt = self._build_individual_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )
        else:
            prompt = self._build_group_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )

        # 尝试多次生成，直到成功或达到最大重试次数
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response = create_chat_completion(
                    self.client,
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(is_individual)},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1),  # 每次重试降低温度
                    # 不设置max_tokens，让LLM自由发挥
                )
                
                content = extract_chat_completion_text(response)
                
                # 检查是否被截断（finish_reason不是'stop'）
                finish_reason = response.choices[0].finish_reason
                if finish_reason == 'length':
                    logger.warning(f"LLM输出被截断 (attempt {attempt+1}), 尝试修复...")
                    content = self._fix_truncated_json(content)
                
                # 尝试解析JSON
                try:
                    result = json.loads(content)
                    
                    # 验证必需字段
                    if "bio" not in result or not result["bio"]:
                        result["bio"] = entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}"
                    if "persona" not in result or not result["persona"]:
                        result["persona"] = entity_summary or f"{entity_name}是一个{entity_type}。"
                    
                    return result
                    
                except json.JSONDecodeError as je:
                    logger.warning(f"JSON解析失败 (attempt {attempt+1}): {str(je)[:80]}")
                    
                    # 尝试修复JSON
                    result = self._try_fix_json(content, entity_name, entity_type, entity_summary)
                    if result.get("_fixed"):
                        del result["_fixed"]
                        return result
                    
                    last_error = je
                    
            except Exception as e:
                logger.warning(f"LLM调用失败 (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(1 * (attempt + 1))  # 指数退避
        
        logger.warning(f"LLM生成人设失败（{max_attempts}次尝试）: {last_error}, 使用规则生成")
        result = self._generate_profile_rule_based(
            entity_name, entity_type, entity_summary, entity_attributes
        )
        # 标记规则回退，便于溯源记录真实生成方式
        result["_rule_based_fallback"] = True
        return result
    
    def _fix_truncated_json(self, content: str) -> str:
        """修复被截断的JSON（输出被max_tokens限制截断）"""
        import re
        
        # 如果JSON被截断，尝试闭合它
        content = content.strip()
        
        # 计算未闭合的括号
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        # 检查是否有未闭合的字符串
        # 简单检查：如果最后一个引号后没有逗号或闭合括号，可能是字符串被截断
        if content and content[-1] not in '",}]':
            # 尝试闭合字符串
            content += '"'
        
        # 闭合括号
        content += ']' * open_brackets
        content += '}' * open_braces
        
        return content
    
    def _try_fix_json(self, content: str, entity_name: str, entity_type: str, entity_summary: str = "") -> Dict[str, Any]:
        """尝试修复损坏的JSON"""
        import re
        
        # 1. 首先尝试修复被截断的情况
        content = self._fix_truncated_json(content)
        
        # 2. 尝试提取JSON部分
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()
            
            # 3. 处理字符串中的换行符问题
            # 找到所有字符串值并替换其中的换行符
            def fix_string_newlines(match):
                s = match.group(0)
                # 替换字符串内的实际换行符为空格
                s = s.replace('\n', ' ').replace('\r', ' ')
                # 替换多余空格
                s = re.sub(r'\s+', ' ', s)
                return s
            
            # 匹配JSON字符串值
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string_newlines, json_str)
            
            # 4. 尝试解析
            try:
                result = json.loads(json_str)
                result["_fixed"] = True
                return result
            except json.JSONDecodeError as e:
                # 5. 如果还是失败，尝试更激进的修复
                try:
                    # 移除所有控制字符
                    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                    # 替换所有连续空白
                    json_str = re.sub(r'\s+', ' ', json_str)
                    result = json.loads(json_str)
                    result["_fixed"] = True
                    return result
                except:
                    pass
        
        # 6. 尝试从内容中提取部分信息
        bio_match = re.search(r'"bio"\s*:\s*"([^"]*)"', content)
        persona_match = re.search(r'"persona"\s*:\s*"([^"]*)', content)  # 可能被截断
        
        bio = bio_match.group(1) if bio_match else (entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}")
        persona = persona_match.group(1) if persona_match else (entity_summary or f"{entity_name}是一个{entity_type}。")
        
        # 如果提取到了有意义的内容，标记为已修复
        if bio_match or persona_match:
            logger.info(f"从损坏的JSON中提取了部分信息")
            return {
                "bio": bio,
                "persona": persona,
                "_fixed": True
            }
        
        # 7. 完全失败，返回基础结构
        logger.warning(f"JSON修复失败，返回基础结构")
        return {
            "bio": entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}",
            "persona": entity_summary or f"{entity_name}是一个{entity_type}。"
        }
    
    def _get_system_prompt(self, is_individual: bool) -> str:
        """获取系统提示词"""
        if is_english_forced():
            base_prompt = (
                "You are an expert social-media persona writer. Generate a "
                "detailed, realistic persona for opinion simulation that "
                "faithfully reflects the real-world facts provided in the user "
                "message. You MUST return valid JSON only, with no text "
                "outside the JSON object and no unescaped newlines inside "
                "string values."
            )
            return f"{base_prompt}\n\n{get_language_instruction()}"
        base_prompt = "你是社交媒体用户画像生成专家。生成详细、真实的人设用于舆论模拟,最大程度还原已有现实情况。必须返回有效的JSON格式，所有字符串值不能包含未转义的换行符。"
        return f"{base_prompt}\n\n{get_language_instruction()}"
    
    def _build_individual_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """构建个人实体的详细人设提示词"""
        
        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "无"
        context_str = context[:self.CONTEXT_PROMPT_CHAR_LIMIT] if context else "无额外上下文"

        if is_english_forced():
            attrs_en = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "none"
            context_en = context[:self.CONTEXT_PROMPT_CHAR_LIMIT] if context else "No additional context"
            return f"""Generate a detailed social-media user persona that faithfully reflects the real-world information available.

Entity name: {entity_name}
Entity type: {entity_type}
Entity summary: {entity_summary}
Entity attributes: {attrs_en}

## Graph context (source facts)
{context_en}

{ENGLISH_GROUNDING_RULES}

Generate JSON with the following fields:

1. bio: social-media profile description, about 200 characters
2. persona: a detailed persona description (about 2000 characters of continuous plain text) covering:
   - Basic information (age, occupation, education background, location) — only as supported by the source facts
   - Background (important experiences, connection to the event, social relationships)
   - Personality (MBTI type, core traits, emotional expression style)
   - Social-media behavior (posting frequency, content preferences, interaction style, language style)
   - Positions and viewpoints (attitude toward the topic, what might anger or move them)
   - Distinctive traits (catchphrases, hobbies, unusual experiences)
   - Personal memory (their connection to the event and actions or reactions they already took, as recorded in the source facts)
3. age: age as an integer (only derivable from the source facts; otherwise choose a plausible adult age)
4. gender: MUST be the English string "male" or "female"
5. mbti: MBTI type (e.g., INTJ, ENFP)
6. country: country name in English (e.g., "United States")
7. profession: occupation (must match the source facts when stated)
8. interested_topics: array of topic strings

Important:
- All field values must be strings or numbers; do not use newline characters
- persona must be one continuous text passage
- Respond in English only (gender must be "male"/"female"; country must be in English)
- Content must stay consistent with the graph context (source facts) above
- age must be a valid integer; gender must be "male" or "female"
"""

        return f"""为实体生成详细的社交媒体用户人设,最大程度还原已有现实情况。

实体名称: {entity_name}
实体类型: {entity_type}
实体摘要: {entity_summary}
实体属性: {attrs_str}

上下文信息（图谱事实，人设的唯一事实依据）:
{context_str}

请生成JSON，包含以下字段:

1. bio: 社交媒体简介，200字
2. persona: 详细人设描述（2000字的纯文本），需包含:
   - 基本信息（年龄、职业、教育背景、所在地）
   - 人物背景（重要经历、与事件的关联、社会关系）
   - 性格特征（MBTI类型、核心性格、情绪表达方式）
   - 社交媒体行为（发帖频率、内容偏好、互动风格、语言特点）
   - 立场观点（对话题的态度、可能被激怒/感动的内容）
   - 独特特征（口头禅、特殊经历、个人爱好）
   - 个人记忆（人设的重要部分，要介绍这个个体与事件的关联，以及这个个体在事件中的已有动作与反应）
3. age: 年龄数字（必须是整数）
4. gender: 性别，必须是英文: "male" 或 "female"
5. mbti: MBTI类型（如INTJ、ENFP等）
6. country: 国家（使用中文，如"中国"）
7. profession: 职业
8. interested_topics: 感兴趣话题数组

重要:
- 所有字段值必须是字符串或数字，不要使用换行符
- persona必须是一段连贯的文字描述
- {get_language_instruction()} (gender字段必须用英文male/female)
- 严禁编造：bio和persona中的事实性内容（身份、经历、关系、事件参与等）只能来自上述图谱事实；缺失的信息必须泛化描述，不得虚构具体细节
- 内容要与实体信息保持一致
- age必须是有效的整数，gender必须是"male"或"female"
"""

    def _build_group_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """构建群体/机构实体的详细人设提示词"""
        
        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "无"
        context_str = context[:self.CONTEXT_PROMPT_CHAR_LIMIT] if context else "无额外上下文"

        if is_english_forced():
            attrs_en = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "none"
            context_en = context[:self.CONTEXT_PROMPT_CHAR_LIMIT] if context else "No additional context"
            return f"""Generate a detailed social-media account setup for an organization or group entity that faithfully reflects the real-world information available.

Entity name: {entity_name}
Entity type: {entity_type}
Entity summary: {entity_summary}
Entity attributes: {attrs_en}

## Graph context (source facts)
{context_en}

{ENGLISH_GROUNDING_RULES}

Generate JSON with the following fields:

1. bio: official account description, about 200 characters, professional tone
2. persona: a detailed account setup description (about 2000 characters of continuous plain text) covering:
   - Organization basics (formal name, nature, background, main functions)
   - Account positioning (account type, target audience, core purpose)
   - Communication style (language style, common expressions, topics to avoid)
   - Content behavior (content types, posting frequency, active hours)
   - Positions and attitudes (official stance on core topics, how controversies are handled)
   - Special notes (the community it represents, operating habits)
   - Organizational memory (its connection to the event and actions or reactions already taken, as recorded in the source facts)
3. age: fixed value 30 (virtual age for an organizational account)
4. gender: fixed value "other"
5. mbti: MBTI type describing the account style (e.g., ISTJ for formal and conservative)
6. country: country name in English (e.g., "United States")
7. profession: description of the organization's function
8. interested_topics: array of focus-area strings

Important:
- All field values must be strings or numbers; no null values
- persona must be one continuous text passage without newline characters
- Respond in English only (gender must be "other"; country must be in English)
- Content must stay consistent with the graph context (source facts) above
- age must be the integer 30 and gender must be the string "other"
- The account's voice must fit the entity's role"""

        return f"""为机构/群体实体生成详细的社交媒体账号设定,最大程度还原已有现实情况。

实体名称: {entity_name}
实体类型: {entity_type}
实体摘要: {entity_summary}
实体属性: {attrs_str}

上下文信息（图谱事实，账号设定的唯一事实依据）:
{context_str}

请生成JSON，包含以下字段:

1. bio: 官方账号简介，200字，专业得体
2. persona: 详细账号设定描述（2000字的纯文本），需包含:
   - 机构基本信息（正式名称、机构性质、成立背景、主要职能）
   - 账号定位（账号类型、目标受众、核心功能）
   - 发言风格（语言特点、常用表达、禁忌话题）
   - 发布内容特点（内容类型、发布频率、活跃时间段）
   - 立场态度（对核心话题的官方立场、面对争议的处理方式）
   - 特殊说明（代表的群体画像、运营习惯）
   - 机构记忆（机构人设的重要部分，要介绍这个机构与事件的关联，以及这个机构在事件中的已有动作与反应）
3. age: 固定填30（机构账号的虚拟年龄）
4. gender: 固定填"other"（机构账号使用other表示非个人）
5. mbti: MBTI类型，用于描述账号风格，如ISTJ代表严谨保守
6. country: 国家（使用中文，如"中国"）
7. profession: 机构职能描述
8. interested_topics: 关注领域数组

重要:
- 所有字段值必须是字符串或数字，不允许null值
- persona必须是一段连贯的文字描述，不要使用换行符
- {get_language_instruction()} (gender字段必须用英文"other")
- 严禁编造：bio和persona中的事实性内容（机构性质、立场、事件参与等）只能来自上述图谱事实；缺失的信息必须泛化描述，不得虚构具体细节
- age必须是整数30，gender必须是字符串"other"
- 机构账号发言要符合其身份定位"""
    
    def _generate_profile_rule_based(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用规则生成基础人设"""
        
        # 根据实体类型生成不同的人设
        entity_type_lower = entity_type.lower()
        
        if entity_type_lower in ["student", "alumni"]:
            return {
                "bio": f"{entity_type} with interests in academics and social issues.",
                "persona": f"{entity_name} is a {entity_type.lower()} who is actively engaged in academic and social discussions. They enjoy sharing perspectives and connecting with peers.",
                "age": random.randint(18, 30),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": "Student",
                "interested_topics": ["Education", "Social Issues", "Technology"],
            }
        
        elif entity_type_lower in ["publicfigure", "expert", "faculty"]:
            return {
                "bio": f"Expert and thought leader in their field.",
                "persona": f"{entity_name} is a recognized {entity_type.lower()} who shares insights and opinions on important matters. They are known for their expertise and influence in public discourse.",
                "age": random.randint(35, 60),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(["ENTJ", "INTJ", "ENTP", "INTP"]),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_attributes.get("occupation", "Expert"),
                "interested_topics": ["Politics", "Economics", "Culture & Society"],
            }
        
        elif entity_type_lower in ["mediaoutlet", "socialmediaplatform"]:
            return {
                "bio": f"Official account for {entity_name}. News and updates.",
                "persona": f"{entity_name} is a media entity that reports news and facilitates public discourse. The account shares timely updates and engages with the audience on current events.",
                "age": 30,  # 机构虚拟年龄
                "gender": "other",  # 机构使用other
                "mbti": "ISTJ",  # 机构风格：严谨保守
                "country": "中国",
                "profession": "Media",
                "interested_topics": ["General News", "Current Events", "Public Affairs"],
            }
        
        elif entity_type_lower in ["university", "governmentagency", "ngo", "organization"]:
            return {
                "bio": f"Official account of {entity_name}.",
                "persona": f"{entity_name} is an institutional entity that communicates official positions, announcements, and engages with stakeholders on relevant matters.",
                "age": 30,  # 机构虚拟年龄
                "gender": "other",  # 机构使用other
                "mbti": "ISTJ",  # 机构风格：严谨保守
                "country": "中国",
                "profession": entity_type,
                "interested_topics": ["Public Policy", "Community", "Official Announcements"],
            }
        
        else:
            # 默认人设
            return {
                "bio": entity_summary[:150] if entity_summary else f"{entity_type}: {entity_name}",
                "persona": entity_summary or f"{entity_name} is a {entity_type.lower()} participating in social discussions.",
                "age": random.randint(25, 50),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_type,
                "interested_topics": ["General", "Social Issues"],
            }
    
    def set_graph_id(self, graph_id: str):
        """设置图谱ID用于Zep检索"""
        self.graph_id = graph_id
    
    def generate_profiles_from_entities(
        self,
        entities: List[EntityNode],
        use_llm: bool = True,
        progress_callback: Optional[callable] = None,
        graph_id: Optional[str] = None,
        parallel_count: int = 5,
        realtime_output_path: Optional[str] = None,
        output_platform: str = "reddit",
        provenance_output_path: Optional[str] = None
    ) -> List[OasisAgentProfile]:
        """
        批量从实体生成Agent Profile（支持并行生成）
        
        Args:
            entities: 实体列表
            use_llm: 是否使用LLM生成详细人设
            progress_callback: 进度回调函数 (current, total, message)
            graph_id: 图谱ID，用于Zep检索获取更丰富上下文
            parallel_count: 并行生成数量，默认5
            realtime_output_path: 实时写入的文件路径（如果提供，每生成一个就写入一次）
            output_platform: 输出平台格式 ("reddit" 或 "twitter")
            provenance_output_path: 溯源文件路径（如果提供，记录每个人设的图谱事实依据）
            
        Returns:
            Agent Profile列表
        """
        import concurrent.futures
        from threading import Lock
        
        # 设置graph_id用于Zep检索
        if graph_id:
            self.graph_id = graph_id
        
        total = len(entities)
        profiles = [None] * total  # 预分配列表保持顺序
        completed_count = [0]  # 使用列表以便在闭包中修改
        provenance_by_user_id: Dict[int, Dict[str, Any]] = {}
        lock = Lock()
        
        # 实时写入文件的辅助函数
        def save_profiles_realtime():
            """实时保存已生成的 profiles 到文件"""
            if not realtime_output_path:
                return
            
            with lock:
                # 过滤出已生成的 profiles
                existing_profiles = [p for p in profiles if p is not None]
                if not existing_profiles:
                    return
                
                try:
                    if output_platform == "reddit":
                        # Reddit JSON 格式
                        profiles_data = [p.to_reddit_format() for p in existing_profiles]
                        with open(realtime_output_path, 'w', encoding='utf-8') as f:
                            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                    else:
                        # Twitter CSV 格式
                        import csv
                        profiles_data = [p.to_twitter_format() for p in existing_profiles]
                        if profiles_data:
                            fieldnames = list(profiles_data[0].keys())
                            with open(realtime_output_path, 'w', encoding='utf-8', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(profiles_data)
                except Exception as e:
                    logger.warning(f"实时保存 profiles 失败: {e}")

        # 实时写入溯源文件的辅助函数（已持有lock时调用）
        def record_provenance(user_id: int, provenance: Dict[str, Any]):
            if not provenance_output_path:
                return
            provenance_by_user_id[user_id] = provenance
            try:
                with open(provenance_output_path, 'w', encoding='utf-8') as f:
                    json.dump(provenance_by_user_id, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"实时保存 persona provenance 失败: {e}")
        
        # Capture locale before spawning thread pool workers
        current_locale = get_locale()

        def generate_single_profile(idx: int, entity: EntityNode) -> tuple:
            """生成单个profile的工作函数"""
            set_locale(current_locale)
            entity_type = entity.get_entity_type() or "Entity"
            
            try:
                profile = self.generate_profile_from_entity(
                    entity=entity,
                    user_id=idx,
                    use_llm=use_llm
                )
                
                # 实时输出生成的人设到控制台和日志
                self._print_generated_profile(entity.name, entity_type, profile)
                
                return idx, profile, None
                
            except Exception as e:
                logger.error(f"生成实体 {entity.name} 的人设失败: {str(e)}")
                # 创建一个基础profile（仍带最小溯源信息）
                fallback_provenance = PersonaProvenance(
                    entity_uuid=entity.uuid,
                    entity_name=entity.name,
                    entity_type=entity_type,
                    llm_used=False,
                    related_edge_count=len(entity.related_edges or []),
                    error=str(e),
                ).to_dict()
                fallback_profile = OasisAgentProfile(
                    user_id=idx,
                    user_name=self._generate_username(entity.name),
                    name=entity.name,
                    bio=f"{entity_type}: {entity.name}",
                    persona=entity.summary or f"A participant in social discussions.",
                    source_entity_uuid=entity.uuid,
                    source_entity_type=entity_type,
                    provenance=fallback_provenance,
                )
                return idx, fallback_profile, str(e)
        
        logger.info(f"开始并行生成 {total} 个Agent人设（并行数: {parallel_count}）...")
        print(f"\n{'='*60}")
        print(f"开始生成Agent人设 - 共 {total} 个实体，并行数: {parallel_count}")
        print(f"{'='*60}\n")
        
        # 使用线程池并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_count) as executor:
            # 提交所有任务
            future_to_entity = {
                executor.submit(generate_single_profile, idx, entity): (idx, entity)
                for idx, entity in enumerate(entities)
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_entity):
                idx, entity = future_to_entity[future]
                entity_type = entity.get_entity_type() or "Entity"
                
                try:
                    result_idx, profile, error = future.result()
                    profiles[result_idx] = profile
                    
                    with lock:
                        completed_count[0] += 1
                        current = completed_count[0]
                        record_provenance(result_idx, profile.provenance)
                    
                    # 实时写入文件
                    save_profiles_realtime()
                    
                    if progress_callback:
                        progress_callback(
                            current, 
                            total, 
                            f"已完成 {current}/{total}: {entity.name}（{entity_type}）"
                        )
                    
                    if error:
                        logger.warning(f"[{current}/{total}] {entity.name} 使用备用人设: {error}")
                    else:
                        logger.info(f"[{current}/{total}] 成功生成人设: {entity.name} ({entity_type})")
                        
                except Exception as e:
                    logger.error(f"处理实体 {entity.name} 时发生异常: {str(e)}")
                    with lock:
                        completed_count[0] += 1
                        fallback_profile = OasisAgentProfile(
                            user_id=idx,
                            user_name=self._generate_username(entity.name),
                            name=entity.name,
                            bio=f"{entity_type}: {entity.name}",
                            persona=entity.summary or "A participant in social discussions.",
                            source_entity_uuid=entity.uuid,
                            source_entity_type=entity_type,
                            provenance=PersonaProvenance(
                                entity_uuid=entity.uuid,
                                entity_name=entity.name,
                                entity_type=entity_type,
                                llm_used=False,
                                related_edge_count=len(entity.related_edges or []),
                                error=str(e),
                            ).to_dict(),
                        )
                        record_provenance(idx, fallback_profile.provenance)
                    profiles[idx] = fallback_profile
                    # 实时写入文件（即使是备用人设）
                    save_profiles_realtime()
        
        print(f"\n{'='*60}")
        print(f"人设生成完成！共生成 {len([p for p in profiles if p])} 个Agent")
        print(f"{'='*60}\n")
        
        return profiles
    
    def _print_generated_profile(self, entity_name: str, entity_type: str, profile: OasisAgentProfile):
        """实时输出生成的人设到控制台（完整内容，不截断）"""
        separator = "-" * 70
        
        # 构建完整输出内容（不截断）
        topics_str = ', '.join(profile.interested_topics) if profile.interested_topics else '无'
        
        output_lines = [
            f"\n{separator}",
            t('progress.profileGenerated', name=entity_name, type=entity_type),
            f"{separator}",
            f"用户名: {profile.user_name}",
            f"",
            f"【简介】",
            f"{profile.bio}",
            f"",
            f"【详细人设】",
            f"{profile.persona}",
            f"",
            f"【基本属性】",
            f"年龄: {profile.age} | 性别: {profile.gender} | MBTI: {profile.mbti}",
            f"职业: {profile.profession} | 国家: {profile.country}",
            f"兴趣话题: {topics_str}",
            separator
        ]
        
        output = "\n".join(output_lines)
        
        # 只输出到控制台（避免重复，logger不再输出完整内容）
        print(output)
    
    def save_profiles(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """
        保存Profile到文件（根据平台选择正确格式）
        
        OASIS平台格式要求：
        - Twitter: CSV格式
        - Reddit: JSON格式
        
        Args:
            profiles: Profile列表
            file_path: 文件路径
            platform: 平台类型 ("reddit" 或 "twitter")
        """
        if platform == "twitter":
            self._save_twitter_csv(profiles, file_path)
        else:
            self._save_reddit_json(profiles, file_path)
    
    def _save_twitter_csv(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        保存Twitter Profile为CSV格式（符合OASIS官方要求）
        
        OASIS Twitter要求的CSV字段：
        - user_id: 用户ID（根据CSV顺序从0开始）
        - name: 用户真实姓名
        - username: 系统中的用户名
        - user_char: 详细人设描述（注入到LLM系统提示中，指导Agent行为）
        - description: 简短的公开简介（显示在用户资料页面）
        
        user_char vs description 区别：
        - user_char: 内部使用，LLM系统提示，决定Agent如何思考和行动
        - description: 外部显示，其他用户可见的简介
        """
        import csv
        
        # 确保文件扩展名是.csv
        if not file_path.endswith('.csv'):
            file_path = file_path.replace('.json', '.csv')
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入OASIS要求的表头
            headers = ['user_id', 'name', 'username', 'user_char', 'description']
            writer.writerow(headers)
            
            # 写入数据行
            for idx, profile in enumerate(profiles):
                # user_char: 完整人设（bio + persona），用于LLM系统提示
                user_char = profile.bio
                if profile.persona and profile.persona != profile.bio:
                    user_char = f"{profile.bio} {profile.persona}"
                # 处理换行符（CSV中用空格替代）
                user_char = user_char.replace('\n', ' ').replace('\r', ' ')
                
                # description: 简短简介，用于外部显示
                description = profile.bio.replace('\n', ' ').replace('\r', ' ')
                
                row = [
                    idx,                    # user_id: 从0开始的顺序ID
                    profile.name,           # name: 真实姓名
                    profile.user_name,      # username: 用户名
                    user_char,              # user_char: 完整人设（内部LLM使用）
                    description             # description: 简短简介（外部显示）
                ]
                writer.writerow(row)
        
        logger.info(f"已保存 {len(profiles)} 个Twitter Profile到 {file_path} (OASIS CSV格式)")
    
    def _normalize_gender(self, gender: Optional[str]) -> str:
        """
        标准化gender字段为OASIS要求的英文格式
        
        OASIS要求: male, female, other
        """
        if not gender:
            return "other"
        
        gender_lower = gender.lower().strip()
        
        # 中文映射
        gender_map = {
            "男": "male",
            "女": "female",
            "机构": "other",
            "其他": "other",
            # 英文已有
            "male": "male",
            "female": "female",
            "other": "other",
        }
        
        return gender_map.get(gender_lower, "other")
    
    def _save_reddit_json(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        保存Reddit Profile为JSON格式
        
        使用与 to_reddit_format() 一致的格式，确保 OASIS 能正确读取。
        必须包含 user_id 字段，这是 OASIS agent_graph.get_agent() 匹配的关键！
        
        必需字段：
        - user_id: 用户ID（整数，用于匹配 initial_posts 中的 poster_agent_id）
        - username: 用户名
        - name: 显示名称
        - bio: 简介
        - persona: 详细人设
        - age: 年龄（整数）
        - gender: "male", "female", 或 "other"
        - mbti: MBTI类型
        - country: 国家
        """
        data = []
        for idx, profile in enumerate(profiles):
            # 使用与 to_reddit_format() 一致的格式
            item = {
                "user_id": profile.user_id if profile.user_id is not None else idx,  # 关键：必须包含 user_id
                "username": profile.user_name,
                "name": profile.name,
                "bio": profile.bio[:150],
                "persona": profile.persona,
                "karma": profile.karma if profile.karma else 1000,
                "created_at": profile.created_at,
                # OASIS必需字段 - 确保都有默认值
                "age": profile.age if profile.age else 30,
                "gender": self._normalize_gender(profile.gender),
                "mbti": profile.mbti if profile.mbti else "ISTJ",
                "country": profile.country if profile.country else "中国",
            }
            
            # 可选字段
            if profile.profession:
                item["profession"] = profile.profession
            if profile.interested_topics:
                item["interested_topics"] = profile.interested_topics
            
            data.append(item)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已保存 {len(profiles)} 个Reddit Profile到 {file_path} (JSON格式，包含user_id字段)")
    
    # 保留旧方法名作为别名，保持向后兼容
    def save_profiles_to_json(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """[已废弃] 请使用 save_profiles() 方法"""
        logger.warning("save_profiles_to_json已废弃，请使用save_profiles方法")
        self.save_profiles(profiles, file_path, platform)
