"""
实体质量过滤器：在人设生成之前过滤垃圾实体

针对OpenZep本地图谱提取中常见的垃圾实体，在生成Agent Profile前做
确定性（规则式）清洗，避免以下实体被误当成社媒发言主体：

1. 法规/法律文件（statutes、regulations、named acts）
2. 被错标成"人物"的产品或公司——产品直接丢弃；公司重新标注为
   Organization保留（合法公司仍是合法发言主体）
3. 样板/文档碎片（文件名、URL、称谓、占位符、数字开头的片段、
   小写开头的未分类通用实体）
4. 空名称实体
5. 低信息实体（无摘要、无属性、无任何关联边——没有任何可依据
   的事实来构建人设）

规则刻意保持保守、可解释、可审计：所有删除都附带原因并记录在
质量报告中，方便人工复核。

通过环境变量 MIROFISH_ENTITY_QUALITY_FILTER=0 可整体关闭。
"""

import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Set

from ..utils.logger import get_logger
from .zep_entity_reader import EntityNode

logger = get_logger('mirofish.entity_quality_filter')

FILTER_ENABLED_ENV = "MIROFISH_ENTITY_QUALITY_FILTER"

# 人物类标签（个人/角色）。被错标为公司或产品的"人物"需要纠正。
PERSON_LABELS: Set[str] = {
    "person", "student", "alumni", "professor", "publicfigure", "expert",
    "faculty", "official", "journalist", "activist", "doctor", "lawyer",
    "employee", "executive", "ceo", "clinicianoperator", "partnernegotiator",
    "telehealthfounder", "healthcareprofessional", "officialspokesperson",
}

# 组织/机构类标签。合法公司保持原样，不被产品规则误伤。
ORGANIZATION_LABELS: Set[str] = {
    "organization", "company", "corporation", "university", "school",
    "governmentagency", "ngo", "mediaoutlet", "institution", "group",
    "community", "pharmacy", "hospital", "agency", "socialmediaplatform",
    "telehealthcompany", "clinic", "clinicchain",
}

# 法规/法律文件类标签——法规不能在社媒上发言。
STATUTE_LABELS: Set[str] = {
    "statute", "regulation", "law", "bill", "legislation", "act",
    "legal", "legaldocument", "rule", "compliancepolicy",
}

# 通用（未分类）标签——OpenZep本地提取的默认标签。
GENERIC_LABELS: Set[str] = {"entity", "node", "extractedentity", ""}

# 法规名称特征（作用于实体名称，避免误伤仅在摘要中讨论法规的主体）。
# "X Act"/"X Law" 用双词组匹配，避免误伤以Act开头的公司名。
_STATUTE_NAME_PATTERNS = [
    re.compile(r"\bstatute\b", re.IGNORECASE),
    re.compile(r"\banti[- ]?kickback\b", re.IGNORECASE),
    re.compile(r"\bpublic law\b", re.IGNORECASE),
    re.compile(r"\bcode of federal regulations\b", re.IGNORECASE),
    re.compile(r"\bc\.f\.r\b", re.IGNORECASE),
    re.compile(r"\bu\.?s\.?c\b", re.IGNORECASE),
    re.compile(r"\bhipaa\b", re.IGNORECASE),
    re.compile(r"\bekra\b", re.IGNORECASE),
    re.compile(r"\baks\b", re.IGNORECASE),
    re.compile(r"\b(?:section|§)\s*\d", re.IGNORECASE),
    re.compile(r"\b(?:title|chapter)\s+\d+", re.IGNORECASE),
    re.compile(r"\b\w+ act\b(?:\s+of\s+\d{4})?", re.IGNORECASE),
]

# 公司名称特征（用于把错标为人物的公司重新标注为Organization）。
_COMPANY_NAME_PATTERN = re.compile(
    r"\b(?:inc|llc|ltd|corp|corporation|gmbh|plc|llp|incorporated|"
    r"limited|company|companies)\b",
    re.IGNORECASE,
)

# 产品名称特征（剂型/浓度/复方等）——产品不是社媒发言主体。
_PRODUCT_NAME_PATTERNS = [
    re.compile(
        r"\b(?:cream|ointment|gel|capsule|capsules|tablet|tablets|troche|"
        r"troches|injection|solution|spray|syrup|suppository|supplement|"
        r"supplements|pill|pills|sachet|lozenge|tincture|patch)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|iu|%)\b", re.IGNORECASE),
    re.compile(r"\bcompounded\b", re.IGNORECASE),
    re.compile(r"\bodt\b", re.IGNORECASE),
]

# 产品摘要特征（仅对错标为人物的实体使用）。
_PRODUCT_SUMMARY_PATTERN = re.compile(
    r"\bis an? (?:product|supplement|medication|drug|beverage|shake|kit|"
    r"pricing plan|subscription plan|tier)\b",
    re.IGNORECASE,
)

# 公司名称摘要特征（仅对错标为人物的实体使用）。
_COMPANY_SUMMARY_PATTERN = re.compile(
    r"\bis an? (?:company|corporation|manufacturer|firm|vendor)\b",
    re.IGNORECASE,
)

# 纯占位符名称。
_PLACEHOLDER_NAMES = {
    "unknown", "n/a", "na", "n.a.", "none", "null", "entity", "user",
    "person", "anonymous", "name", "unnamed", "todo", "test", "placeholder",
}

# 纯称谓/头衔名称（不是人名）。
_HONORIFIC_NAMES = {
    "dr", "mr", "mrs", "ms", "miss", "prof", "professor", "doctor",
    "sir", "madam", "officer", "manager", "ceo", "founder", "owner",
    "coordinator", "nurse", "pharmacist", "provider",
}

# 文件名特征。
_FILENAME_PATTERN = re.compile(
    r"[\w\-+.]+\.(?:pdf|json|csv|txt|md|docx?|xlsx?|pptx?|png|jpe?g|gif|html?)$",
    re.IGNORECASE,
)

# URL/邮箱特征。
_URL_EMAIL_PATTERN = re.compile(
    r"^(?:https?://|www\.)|@[\w\-.]+\.\w+",
    re.IGNORECASE,
)


def _normalize_label(entity: EntityNode) -> str:
    entity_type = entity.get_entity_type()
    return (entity_type or "").strip().lower()


def _has_alpha(text: str) -> bool:
    return any(c.isalpha() for c in text)


def _is_statute_name(name: str) -> bool:
    return any(pattern.search(name) for pattern in _STATUTE_NAME_PATTERNS)


def _is_company_name(name: str) -> bool:
    return bool(_COMPANY_NAME_PATTERN.search(name))


def _is_product_name(name: str) -> bool:
    return any(pattern.search(name) for pattern in _PRODUCT_NAME_PATTERNS)


def _has_meaningful_attributes(entity: EntityNode) -> bool:
    for value in (entity.attributes or {}).values():
        if value is not None and str(value).strip():
            return True
    return False


@dataclass
class EntityQualityDecision:
    """单条实体的过滤决策"""
    entity_name: str
    entity_type: Optional[str]
    action: str  # "drop" | "relabel" | "keep"
    reason: str
    relabeled_to: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "action": self.action,
            "reason": self.reason,
            "relabeled_to": self.relabeled_to,
        }


@dataclass
class EntityQualityReport:
    """实体质量过滤结果（含审计信息）"""
    kept: List[EntityNode] = field(default_factory=list)
    dropped: List[EntityQualityDecision] = field(default_factory=list)
    relabeled: List[EntityQualityDecision] = field(default_factory=list)

    @property
    def total_input(self) -> int:
        return len(self.kept) + len(self.dropped) + len(self.relabeled)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kept_count": len(self.kept),
            "dropped_count": len(self.dropped),
            "relabeled_count": len(self.relabeled),
            "total_input": self.total_input,
            "kept": [
                {"name": e.name, "type": e.get_entity_type()}
                for e in self.kept
            ],
            "dropped": [d.to_dict() for d in self.dropped],
            "relabeled": [d.to_dict() for d in self.relabeled],
        }


class EntityQualityFilter:
    """在人设生成之前过滤垃圾实体的确定性过滤器"""

    def _drop(
        self, entity: EntityNode, reason: str, report: EntityQualityReport
    ) -> None:
        report.dropped.append(
            EntityQualityDecision(
                entity_name=entity.name,
                entity_type=entity.get_entity_type(),
                action="drop",
                reason=reason,
            )
        )

    def _relabel_as_organization(
        self, entity: EntityNode, reason: str, report: EntityQualityReport
    ) -> EntityNode:
        new_labels = [
            "Organization" if (label or "").strip().lower() in PERSON_LABELS
            else label
            for label in entity.labels
        ]
        if "Organization" not in new_labels:
            new_labels = ["Organization"] + new_labels
        relabeled = replace(entity, labels=new_labels)
        report.relabeled.append(
            EntityQualityDecision(
                entity_name=entity.name,
                entity_type=entity.get_entity_type(),
                action="relabel",
                reason=reason,
                relabeled_to="Organization",
            )
        )
        return relabeled

    def filter_entity(
        self, entity: EntityNode, report: EntityQualityReport
    ) -> Optional[EntityNode]:
        """
        对单个实体做质量决策。

        返回保留的实体（可能是重新标注后的副本），垃圾实体返回None
        并记录在report中。
        """
        label = _normalize_label(entity)
        name = (entity.name or "").strip()

        # 1. 空名称
        if not name:
            self._drop(entity, "empty_name", report)
            return None

        # 2. 法规/法律文件（按标签或名称判断）
        if label in STATUTE_LABELS or _is_statute_name(name):
            self._drop(entity, "statute", report)
            return None

        # 3. 人物类实体：公司/产品错标纠正
        if label in PERSON_LABELS:
            if _is_company_name(name) or (
                entity.summary and _COMPANY_SUMMARY_PATTERN.search(entity.summary)
            ):
                return self._relabel_as_organization(
                    entity, "company_mislabeled_as_person", report
                )
            if _is_product_name(name) or (
                entity.summary and _PRODUCT_SUMMARY_PATTERN.search(entity.summary)
            ):
                self._drop(entity, "product_mislabeled_as_person", report)
                return None

        # 4. 非组织类实体的产品名（剂型/浓度等）——产品不是发言主体。
        #    组织类标签优先豁免，避免误伤合法公司。
        if label not in ORGANIZATION_LABELS and _is_product_name(name):
            self._drop(entity, "product_not_a_speaker", report)
            return None

        # 5. 样板/文档碎片
        normalized_name = re.sub(r"[^\w]", "", name, flags=re.UNICODE).lower()
        if normalized_name in _PLACEHOLDER_NAMES or name.lower().strip(".·") in _PLACEHOLDER_NAMES:
            self._drop(entity, "placeholder_name", report)
            return None
        if name.lower().rstrip(".") in _HONORIFIC_NAMES:
            self._drop(entity, "honorific_title_only", report)
            return None
        if _FILENAME_PATTERN.search(name):
            self._drop(entity, "filename_fragment", report)
            return None
        if _URL_EMAIL_PATTERN.search(name):
            self._drop(entity, "url_or_email_fragment", report)
            return None
        if not _has_alpha(name):
            self._drop(entity, "no_alphabetic_characters", report)
            return None
        if len(re.findall(r"[^\W\d_]", name)) <= 1:
            self._drop(entity, "name_too_short", report)
            return None
        if name[0].isdigit():
            self._drop(entity, "number_led_fragment", report)
            return None

        # 6. 未分类通用实体的小写碎片（OpenZep默认提取的概念片段）
        if label in GENERIC_LABELS and name[0].islower():
            self._drop(entity, "generic_lowercase_fragment", report)
            return None

        # 7. 低信息实体：无摘要、无属性、无边——无法构建有依据的人设
        has_summary = bool((entity.summary or "").strip())
        if (
            not has_summary
            and not _has_meaningful_attributes(entity)
            and not (entity.related_edges or [])
            and not (entity.related_nodes or [])
        ):
            self._drop(entity, "low_information_entity", report)
            return None

        return entity

    def filter_entities(
        self, entities: List[EntityNode]
    ) -> EntityQualityReport:
        """对实体列表做质量过滤，返回保留/剔除/重标详情。"""
        report = EntityQualityReport()
        for entity in entities:
            kept = self.filter_entity(entity, report)
            if kept is not None:
                report.kept.append(kept)
        logger.info(
            f"实体质量过滤完成: 输入 {report.total_input} 个, 保留 {len(report.kept)} 个, "
            f"剔除 {len(report.dropped)} 个, 重标 {len(report.relabeled)} 个"
        )
        return report


def is_entity_quality_filter_enabled() -> bool:
    """质量过滤默认开启；MIROFISH_ENTITY_QUALITY_FILTER=0/false 可关闭。"""
    value = (os.environ.get(FILTER_ENABLED_ENV, "") or "").strip().lower()
    return value not in {"0", "false", "off", "no"}


def filter_entities_for_profiles(
    entities: List[EntityNode],
) -> EntityQualityReport:
    """
    在人设生成前应用质量过滤（考虑开关）。

    关闭时返回原列表并生成空报告（不修改任何实体）。
    """
    if not is_entity_quality_filter_enabled():
        report = EntityQualityReport(kept=list(entities))
        logger.info("实体质量过滤已通过环境变量关闭，跳过过滤")
        return report
    return EntityQualityFilter().filter_entities(entities)
