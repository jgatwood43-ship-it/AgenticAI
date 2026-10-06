"""
core/investigation_planner.py
─────────────────────────────
Decompose broad OR targeted security/correlation requests into focused,
evidence-oriented investigation tasks.

The planner does not generate SQL.

Each task is later passed independently to SQLPlanner, which produces a
schema-grounded logical query plan and deterministic SQL.

The planner uses four inputs:
    1. User investigative objective
    2. Authoritative schema documentation
    3. Deterministic SchemaGraph tables/relationships derived from that documentation
    4. Retrieved policy / standards evidence

The planner LLM chooses WHAT operational evidence must be investigated.
Physical table selections recorded on tasks are planning metadata; SQLPlanner
receives the complete authoritative schema and independently chooses the physical
representation used for SQL. Python/SchemaGraph validates documented structure.

Task questions must be natural-language, evidence-oriented, and grounded in
documented operational domains. SQL, pseudo-SQL, meta/compliance, unsafe table
references, and invalid SchemaGraph relationships are rejected before they reach
SQLPlanner.

Natural-language evidence concepts and requested_outputs are SEMANTIC labels.
They do not have to be exact physical database column names. SQLPlanner is
responsible for mapping those concepts to documented columns.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any

from llama_index.core.llms import ChatMessage, LLM
from pydantic import BaseModel, Field

from core.schema_graph import RelationshipPath, SchemaGraph
from core.system_prompts import INVESTIGATION_PLANNER_SYSTEM_PROMPT


def _complete_with_system(
    llm: Any,
    *,
    system_prompt: str,
    user_prompt: str,
) -> Any:
    """Run an LLM turn with an explicit system-role reasoning contract."""
    return llm.chat(
        [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
    )


MAX_INVESTIGATION_TASKS = 16
MAX_INITIAL_TARGETED_PLAN_ATTEMPTS = 3
MAX_SUPPLEMENTAL_PLAN_ATTEMPTS = 3


class InvestigationScope(str, Enum):
    ACCESS_AUTHORIZATION = "access_authorization"
    EMPLOYMENT_STATUS = "employment_status"
    ATTENDANCE_STATE = "attendance_state"
    AFTER_HOURS_ACCESS = "after_hours_access"
    DUAL_CUSTODY = "dual_custody"
    KEY_ACTIVITY = "key_activity"
    VIDEO_CORRELATION = "video_correlation"
    DENIED_ACCESS = "denied_access"
    GENERAL = "general"


class InvestigationTask(BaseModel):
    """One focused analysis that may be planned and executed independently."""

    task_id: str
    title: str
    question: str
    scope: InvestigationScope

    required_tables: list[str] = Field(default_factory=list)

    # Populated deterministically by Python/SchemaGraph.
    relationship_paths: list[str] = Field(default_factory=list)

    # Recomputed by Python from graph-resolved tables.
    required_schema_documents: list[str] = Field(default_factory=list)

    requested_outputs: list[str] = Field(default_factory=list)
    rationale: str
    priority: int = Field(default=3, ge=1, le=5)


class InvestigationPlan(BaseModel):
    """Structured decomposition of one broad or targeted investigation request."""

    purpose: str
    is_investigation: bool
    tasks: list[InvestigationTask] = Field(
        default_factory=list,
        max_length=MAX_INVESTIGATION_TASKS,
    )
    assumptions: list[str] = Field(default_factory=list)
    missing_business_rules: list[str] = Field(default_factory=list)


class SupplementalPlanReview(BaseModel):
    """LLM reflection on whether a supplemental plan actually closes the gap."""

    sufficient: bool
    reason: str
    missing_requirements: list[str] = Field(default_factory=list)
    repeated_evidence_domains: list[str] = Field(default_factory=list)


_INVESTIGATION_PROMPT = """
You are an investigation-planning agent for a cybersecurity evidence-analysis
system.

You do not write SQL.

Your job is to decompose one investigation request into focused,
evidence-oriented NATURAL-LANGUAGE questions that a separate SQLPlanner can
later translate into SQL.

The request may be either:
- BROAD: a multi-area security review; or
- TARGETED: a bounded correlation/comparison question whose answer requires
  evidence from multiple organizational domains or states.

USER REQUEST
------------
{query}

PERSISTENT INVESTIGATION EVIDENCE LEDGER
----------------------------------------
{ledger_context}

The ledger is authoritative for what has already been learned. Semantic or
derived ledger labels are reasoning concepts, NOT database table/column names.

AUTHORITATIVE SCHEMA
--------------------
{schema_context}

SCHEMAGRAPH TABLE CATALOG
-------------------------
{graph_table_catalog}

SCHEMAGRAPH DOCUMENTED EVIDENCE-PLANNING RELATIONSHIPS
-------------------------------------------
{graph_relationship_catalog}

SCHEMAGRAPH RULE
----------------
The SchemaGraph content above is deterministic and authoritative for documented evidence-domain connectivity. Planning relationships may be temporal, indirect, domain, logical, or direct; they validate WHAT evidence domains may be related, but do not certify HOW SQL must join them.

You may choose WHAT tables are relevant by populating task.required_tables.

You MUST NOT choose join columns or relationship paths.
Leave task.relationship_paths empty. Python resolves and validates legal paths.

If the tables needed for an analysis cannot be connected through the documented SchemaGraph planning relationships, omit the task and record the limitation. A planning path proves only documented evidence-domain connectivity; SQLPlanner independently chooses and validates the physical SQL construction.

RETRIEVED POLICY / STANDARDS EVIDENCE
-------------------------------------
{policy_context}

RULES
-----
1. task.required_tables must contain only documented table names.
   Evidence-domain relationship structure must come only from AUTHORITATIVE SCHEMA / SCHEMAGRAPH. A planning relationship is not a certified SQL join.

   task.question and task.requested_outputs are SEMANTIC evidence descriptions.
   They may use normal business concepts such as "employee name", "employee_name",
   "clock state", "access event", or "event timestamp" without those phrases
   needing to be literal database column names.

   Do NOT claim that a semantic label is a physical column. SQLPlanner will map
   the semantic evidence request to documented physical columns later.
2. Use policy evidence only to identify relevant security principles or areas
   worth testing.
3. Do not invent table names, relationships, employees, rooms, dates,
   thresholds, schedules, business values, or findings. Do not invent physical
   columns when explicitly naming a database field.
4. Convert policy principles into factual questions that can be tested from
   available data.
5. Every task.question MUST be normal English prose.
6. NEVER put SQL, pseudo-SQL, query syntax, braces around SQL, subqueries, or
   implementation instructions in task.question.

   Normal English words such as "where", "from", "in", "before", and "after"
   are allowed. Do not avoid natural-language phrasing merely because those
   words also exist in SQL.

7. Describe WHAT evidence should be found, not HOW SQL should retrieve it.
8. Each task should be answerable by one focused database query.
9. Select only table names present in SCHEMAGRAPH TABLE CATALOG.
   Put EVERY table needed to establish the task's condition or requested
   outputs in task.required_tables.

   required_tables is a completeness contract, not merely a starting table.
   For example, if a task compares an event with an employee record, room
   record, authorization rule, key record, time/attendance record, or video
   record, include the documented tables needed for BOTH sides of that
   comparison.

   If task.question contains a cross-domain condition such as one event/state
   occurring while, before, after, or without another event/state, the task is
   incomplete unless required_tables contains the documented data domains needed
   to observe BOTH sides. Otherwise split the work into complementary tasks.

   Do not specify join columns or relationship paths; Python resolves them.
10. Preserve useful factual outputs such as names, timestamps, rooms,
    departments, access results, status, or event details where supported.
11. Avoid duplicate or overlapping tasks.
12. Do not claim an issue exists; ask whether evidence exists.
13. If a concern cannot be tested from schema, record the limitation instead.
14. If a test requires an undocumented business rule, record it in
    missing_business_rules rather than inventing it.
15. Leave task.relationship_paths empty. Python/SchemaGraph populates it.
    required_schema_documents may be left empty because Python recomputes it.
16. requested_outputs must contain factual SEMANTIC outputs, not conclusions
    or SQL. They are evidence labels for SQLPlanner and do NOT need to match
    physical column names one-for-one.
17. Limit the plan to the most useful {max_tasks} tasks.
18. For broad investigations, produce at least one task whenever schema
    supports at least one security-relevant factual analysis.
19. Do not return an empty task list merely because the request is general.

20. Every task must request OPERATIONAL EVIDENCE from records.

21. Do NOT create tasks that ask:
    - what a table means or what its purpose is;
    - what relationships exist in the schema;
    - whether a dataset is sufficient;
    - whether a table supports NIST or CIS compliance;
    - whether a database proves policy compliance;
    - whether schema documentation itself is complete.

22. Schema description is already available from AUTHORITATIVE SCHEMA and does
    not require a database investigation.

23. Policy compliance is interpreted later by the evidence grader. Your job is
    only to identify concrete database evidence that may support or contradict
    relevant security principles.

24. A good task asks for operational records, events, actors, timestamps,
    statuses, results, mismatches, missing references, or inconsistencies.

25. A bad task asks questions such as:
    - "What is this table for?"
    - "What relationships does this table have?"
    - "Is this dataset sufficient?"
    - "Is this table NIST compliant?"
    - "Does this prove CIS compliance?"

26. Prefer simple, well-grounded operational evidence tasks over speculative,
    descriptive, or compliance-meta tasks.

27. An investigation task may gather relevant operational evidence without
    encoding the final security conclusion in the database query.

28. Separate OBSERVATION from INTERPRETATION:
    - the database query gathers documented records and facts;
    - the downstream evidence grader determines whether those facts indicate a
      potential security issue.

29. If a user concept contains both observable and qualitative parts, preserve
    the observable evidence request instead of rejecting the entire task.
    For example, a request involving "missing or invalid" data may still gather
    missing/null/unmatched records even when "invalid" requires an undocumented
    business rule.

30. It is acceptable for an evidence task to retrieve records useful for later
    correlation or reasoning, even when no single returned row proves a
    security issue by itself.

31. Prefer factual evidence such as:
    - operational events and their actors/timestamps/results;
    - authorization/rule records related to those events;
    - employee status/history related to events;
    - room, key, camera, and footage records;
    - missing or unmatched documented references;
    - other documented records that could be compared downstream.

32. Do not ask the database to decide whether something is "secure",
    "suspicious", "risky", "compliant", or a "security issue". Ask for the
    underlying records that would let the downstream LLM evaluate that concept.

33. For broad reviews, produce multiple non-overlapping evidence tasks when the
    documented schema supports them.

34. Coverage should be driven by documented data and relationships, not a
    hard-coded list of security tests.

35. Treat AUTHORITATIVE SCHEMA and SCHEMAGRAPH as given facts, not investigation
    targets.

36. Cross-table evidence tasks must name the endpoint tables needed by the
    evidence request in required_tables. Python/SchemaGraph resolves legal
    relationship paths.

37. requested_outputs should name factual evidence useful to the final answer,
    such as employee name/identity, event timestamp, room/location, access
    result, clock activity/state, status, key, camera, or other evidence concepts.

    These are SEMANTIC output labels. SQLPlanner is responsible for mapping them
    to the best documented physical representation. Do not force a semantic fact
    such as "clock state" into a particular raw-event or summary representation
    unless the task genuinely requires that representation.

38. Do not invent thresholds, schedules, meanings, categorical values, status
    values, result values, direction values, or business rules that are absent
    from the user request, policy evidence, or authoritative schema.

    A varchar/text column name does NOT authorize you to invent one of its
    possible stored values. If allowed values are not documented, retrieve the
    factual column value without filtering on a guessed literal and let the
    downstream LLM interpret the returned evidence.

39. For a TARGETED correlation investigation:
    - identify EVERY factual side/domain that must be compared to answer the
      user's question;
    - produce the smallest useful set of complementary evidence tasks that
      preserves evidence from ALL of those sides/domains;
    - usually 2 to 4 tasks are appropriate when separate operational domains
      must later be correlated;
    - a single task is acceptable ONLY when its required_tables and requested
      outputs genuinely include every side of the requested comparison;
    - do not create a task whose wording claims to test a cross-domain condition
      while its required_tables cover only one side of that condition;
    - preserve shared factual fields needed for downstream correlation, such as
      employee identifiers/names, event timestamps, room/location, status, and
      other documented comparison fields;
    - when separate tasks are used, preserve common correlation keys/fields in
      each task so the downstream LLM can compare the returned evidence;
    - do not expand the request into an unrelated broad security review;
    - do not force one SQL task to make the final security judgment when the
      downstream LLM can compare verified evidence across tasks.

40. For TARGETED correlation questions, explicitly check your proposed plan
    before returning it:
    - Can the successful results, taken together, contain evidence for every
      factual component named in the user's requested comparison?
    - If one side/domain is missing from the tasks, add a complementary evidence
      task before returning the plan.
    - Do not treat access/activity evidence from one domain as evidence about
      another domain's state.

41. For a BROAD investigation, produce multiple distinct evidence tasks when
    supported by the schema.

42. Limit the plan to the most useful {max_tasks} evidence tasks.
"""

_INITIAL_TARGETED_PLAN_REVIEW_PROMPT = """
You are reviewing a proposed INITIAL TARGETED INVESTIGATION PLAN before any SQL
is generated or executed.

ORIGINAL USER QUESTION
----------------------
{query}

PERSISTENT INVESTIGATION EVIDENCE LEDGER
----------------------------------------
{ledger_context}

PROPOSED INVESTIGATION PLAN
---------------------------
{proposed_plan}

AUTHORITATIVE TABLE CATALOG
---------------------------
{graph_table_catalog}

AUTHORITATIVE EVIDENCE-PLANNING RELATIONSHIPS
----------------------------------
{graph_relationship_catalog}

TASK
----
Decide whether the proposed plan, if all tasks execute successfully, can gather
the factual evidence needed to resolve the TARGETED cross-domain question.

Set sufficient=true only when the proposed tasks, TAKEN TOGETHER, cover every
material evidence domain represented by the user's question and the ledger's
still-missing semantic requirements.

MANDATORY REVIEW RULES
----------------------
1. Review the WHOLE PLAN, not each task in isolation.
2. Semantic/derived ledger labels are reasoning concepts, not table or column
   names. Do not require a physical column with the same name.
3. Map semantic requirements to documented raw evidence domains in the table
   catalog.
4. If the ledger says a material domain is still missing and a documented table
   can provide raw evidence for that domain, the plan must contain at least one
   task that retrieves that domain.
5. A targeted cross-domain condition may be handled either by one task whose
   required_tables genuinely cover all sides or by complementary tasks whose
   successful evidence can be correlated later.
6. Prefer complementary factual evidence tasks when that avoids forcing one SQL
   query to make the final semantic/security judgment.
7. Do not accept a plan whose wording claims to test a cross-domain condition
   while the plan's required_tables cover only one side.
8. Do not reject a plan merely because a semantic fact must later be DERIVED
   from raw evidence. The downstream grader can derive facts such as state at a
   timestamp from documented event records.
9. Do not invent table names, columns, relationships, or business rules.
10. If insufficient, identify exactly which semantic/evidence domains remain
    absent and explain what documented raw evidence is needed.
"""


_SUPPLEMENTAL_INVESTIGATION_PROMPT = """
You are planning ONE BOUNDED NEXT-EVIDENCE PASS for a cybersecurity
investigation.

You do not write SQL.

MISSING-EVIDENCE OBJECTIVE — THIS OVERRIDES THE ORIGINAL QUESTION
---------------------------------------------------------------
{supplemental_request}

ORIGINAL USER QUESTION — CONTEXT ONLY
-------------------------------------
{original_query}

ALREADY SUCCESSFUL EVIDENCE METADATA
------------------------------------
{existing_evidence_summary}

PERSISTENT INVESTIGATION EVIDENCE LEDGER
----------------------------------------
{ledger_context}

Use the ledger to avoid re-requesting covered evidence. Semantic/derived ledger
labels are reasoning concepts and MUST NOT be copied as technical identifiers.

AUTHORITATIVE SCHEMA
--------------------
{schema_context}

SCHEMAGRAPH TABLE CATALOG
-------------------------
{graph_table_catalog}

SCHEMAGRAPH DOCUMENTED EVIDENCE-PLANNING RELATIONSHIPS
-------------------------------------------
{graph_relationship_catalog}

RETRIEVED POLICY / STANDARDS EVIDENCE
-------------------------------------
{policy_context}

TASK
----
Create only the smallest set of NEW factual evidence tasks needed to satisfy
the MISSING-EVIDENCE OBJECTIVE above.

The objective describes a REAL-WORLD FACTUAL GAP. Convert that gap into a
question about the missing operational evidence itself.

MANDATORY RULES
---------------
1. The MISSING-EVIDENCE OBJECTIVE is authoritative. It may come from Agent 2's
   post-query sufficiency review or Agent 3's later holistic review. Do not
   re-plan or re-answer the original user question.
1a. NEVER turn diagnostic language about prior evidence into a meta-question.
    In particular, do NOT create tasks such as:
    - "Why is the employee's clock-in status missing?"
    - "Why was evidence not returned?"
    - "Why did the previous query fail?"
    - "Why is a field absent?"

    Instead, ask directly for the missing FACTUAL evidence, for example:
    - "What documented time-clock evidence establishes each employee's clock
      state at the relevant access timestamp?"
    - "What documented records identify the employee associated with each
      relevant event?"

    The examples illustrate the distinction between a factual evidence question
    and a meta-question. Choose the actual evidence task yourself from the
    supplied objective and authoritative schema.
1b. Treat any section labeled DIAGNOSTIC CONTEXT as explanation only. It may help
    you understand the gap, but it must not become task.question.
2. Do not repeat an already successful evidence task unless the primary objective
   explicitly says a missing field must be added to that same evidence domain.
3. If the missing-evidence objective names a missing operational domain (for example,
   time-clock data), at least one returned task MUST retrieve evidence from that
   documented domain when the schema supports it.
4. Preserve common factual correlation keys such as employee identifiers/names,
   timestamps, rooms/locations, statuses, and event details when documented.
5. Ask only for factual operational records. The downstream LLM makes the final
   security judgment.
6. Use only documented table names in required_tables.
7. Include every table needed for the supplemental evidence task, but do not add
   unrelated tables.
8. task.question must be normal English prose. Do not write SQL or pseudo-SQL.
9. Leave relationship_paths empty. Python/SchemaGraph resolves legal paths.
10. requested_outputs must be factual SEMANTIC evidence outputs, not
    conclusions. They do not have to be literal database column names.
11. Do not invent thresholds, schedules, dates, categorical values, status
    values, result values, direction values, or business-rule meanings.
12. A varchar/text column does NOT document its possible values. If values are
    not explicitly supplied by the user or authoritative evidence, request the
    raw column instead of filtering on a guessed literal.
12a. Never convert a semantic ledger concept such as "clock state",
    "authorization state", or "access event" into a guessed column name.
    Retrieve documented raw fields that allow the downstream LLM to derive it.
12b. task.question must ask WHAT operational fact or records should be retrieved,
    not WHY the prior investigation lacked them. If the objective says evidence
    was insufficient because a state was not established, ask for evidence that
    establishes that state.
12c. Preserve the original user's correlation target when planning the missing
    fact. For a state-at-event-time gap, retrieve evidence that can establish the
    state at the relevant event time rather than merely retrieving unrelated
    records from the same domain.
12d. Keep the task REPRESENTATION-NEUTRAL unless the missing-evidence objective
    explicitly requires a particular documented representation. For example,
    request "documented time-clock evidence that establishes clock state" rather
    than prematurely narrowing the task to "time-clock events", raw punches, a
    summary table, or any other physical representation.
12e. When multiple documented representations could plausibly supply the same
    semantic fact, do not choose one merely because its table name resembles the
    business concept. Let SQLPlanner inspect the complete authoritative schema
    and choose the physical representation.
13. Prefer 1 to 3 supplemental tasks. Return no unrelated broad-review tasks.
14. If the requested missing evidence cannot be retrieved from the documented
    schema, return no task for that component and explain the limitation in
    missing_business_rules.
15. Return only the structured InvestigationPlan.
"""


_SUPPLEMENTAL_PLAN_REVIEW_PROMPT = """
You are reviewing a proposed SUPPLEMENTAL EVIDENCE PLAN before any SQL is
generated or executed.

MISSING-EVIDENCE OBJECTIVE
--------------------------
{supplemental_request}

ORIGINAL USER QUESTION — CONTEXT ONLY
-------------------------------------
{original_query}

ALREADY SUCCESSFUL EVIDENCE METADATA
------------------------------------
{existing_evidence_summary}

PERSISTENT INVESTIGATION EVIDENCE LEDGER
----------------------------------------
{ledger_context}

PROPOSED SUPPLEMENTAL PLAN
--------------------------
{proposed_plan}

AUTHORITATIVE TABLE CATALOG
---------------------------
{graph_table_catalog}

TASK
----
Decide whether the proposed next-evidence plan actually retrieves the factual
evidence identified by the current missing-evidence objective.

Set sufficient=true ONLY when the plan, if successfully executed, would add
factual evidence that materially closes the stated evidence gap.

MANDATORY REVIEW RULES
----------------------
1. Compare required_tables, task questions, and requested_outputs directly with
   the missing-evidence objective.
2. If the objective names a missing operational domain and the schema contains a
   documented table for that domain, the plan must include evidence from that
   domain.
3. Do not accept a plan that simply repeats already successful evidence unless
   it adds a specifically missing factual field.
4. Do not accept a task whose wording claims to test a cross-domain condition
   when its tables cover only one side.
5. Review WHAT evidence will be retrieved, not HOW SQL will implement it.
5a. Reject as insufficient any proposed task that asks WHY prior evidence was
    missing, WHY a previous query failed, or WHY a field was absent instead of
    retrieving the missing operational fact itself.
5b. Do not require the plan to commit to raw events, summary records, or another
    physical representation when the objective is semantic (for example, clock
    state at a timestamp). A representation-neutral task is preferable because
    SQLPlanner receives the complete schema and chooses the physical evidence
    source.
6. If insufficient, identify the exact missing requirements and any repeated
   evidence domains.
7. Do not invent schema facts.
"""


_INVESTIGATION_TERMS = (
    "review the records",
    "review company records",
    "review the company records",
    "review the tables",
    "review tables",
    "review for",
    "identify anomalies",
    "find anomalies",
    "security anomalies",
    "possible security issues",
    "potential security issues",
    "security concerns",
    "possible security concerns",
    "potential security concerns",
    "possible security risks",
    "potential security risks",
    "suspicious activity",
    "investigate",
    "review for inconsistencies",
    "assess security",
    "evaluate security",
    "security review",
)

_ANALYTICAL_TERMS = (
    "review",
    "inspect",
    "assess",
    "analyze",
    "analyse",
    "evaluate",
    "compare",
    "correlate",
    "investigate",
    "determine whether",
    "determine if",
    "identify concerns",
    "identify risks",
    "find anomalies",
    "security concern",
    "security issue",
    "security risk",
    "suspicious",
    "anomaly",
    "violation",
    "inconsistency",
)

_POLICY_TERMS = (
    "nist",
    "cis",
    "policy",
    "policies",
    "standard",
    "standards",
    "control",
    "controls",
    "guidance",
)

_DATA_SCOPE_TERMS = (
    "record",
    "records",
    "table",
    "tables",
    "database",
    "data",
    "activity",
    "events",
    "employees",
    "users",
    "access",
    "badge",
    "key",
    "video",
    "clock",
)

_SQL_LIKE_PATTERNS = (
    # SELECT ... FROM ...
    r"\bselect\b[\s\S]{0,300}\bfrom\b",
    # INSERT / UPDATE / DELETE statements
    r"\binsert\s+into\b",
    r"\bupdate\s+[A-Za-z_][A-Za-z0-9_]*\s+set\b",
    r"\bdelete\s+from\b",
    # FROM <identifier> ... JOIN <identifier>
    r"\bfrom\s+[A-Za-z_][A-Za-z0-9_]*(?:\s+(?:as\s+)?[A-Za-z_][A-Za-z0-9_]*)?"
    r"\s+(?:inner\s+|left\s+|right\s+|full\s+|cross\s+)?join\s+"
    r"[A-Za-z_][A-Za-z0-9_]*",
    # WHERE followed by a column-like comparison expression.
    r"\bwhere\s+[A-Za-z_][A-Za-z0-9_.]*\s*"
    r"(?:=|<>|!=|>=|<=|>|<|\bis\b|\blike\b|\bin\b)",
    # Common SQL clause structures.
    r"\bgroup\s+by\s+[A-Za-z_][A-Za-z0-9_.]*",
    r"\border\s+by\s+[A-Za-z_][A-Za-z0-9_.]*",
    r"\bhaving\s+(?:count|sum|avg|min|max)\s*\(",
    # Subquery / set-operation structures.
    r"\bnot\s+in\s*\(\s*select\b",
    r"\bin\s*\(\s*select\b",
    r"\bexists\s*\(\s*select\b",
    r"\bunion(?:\s+all)?\s+select\b",
    r"\bintersect\s+select\b",
    r"\bexcept\s+select\b",
    r"\bwith\s+[A-Za-z_][A-Za-z0-9_]*\s+as\s*\(\s*select\b",
    # Braced pseudo-SQL occasionally generated by the LLM.
    r"\{\s*select\b",
)

_SQL_OUTPUT_PATTERNS = (
    r"\bselect\b",
    r"\bwhere\b",
    r"\bjoin\b",
    r"\bcount\s*\(",
    r"\bsum\s*\(",
    r"\bavg\s*\(",
    r"\bmin\s*\(",
    r"\bmax\s*\(",
)


_META_TASK_PATTERNS = (
    r"\bwhy (?:is|are|was|were|did|does)\b.*\b(?:missing|absent|fail|failed|failure)\b",
    r"\bwhy\b.*\b(?:evidence|query|field|column|status)\b.*\b(?:missing|absent|fail|failed)\b",
    r"\bwhat is the purpose\b",
    r"\bwhat does .* table\b",
    r"\bdescribe (?:the )?.*table\b",
    r"\brelationships? between .* tables?\b",
    r"\brelationships? .* other tables?\b",
    r"\bwhat relationships?\b",
    r"\bsufficient to determine\b",
    r"\bsufficient to support\b",
    r"\bprovide evidence of .* compliance\b",
    r"\bdetermine .* policy compliance\b",
    r"\bpolicy compliance\b",
    r"\bnist .* compliance\b",
    r"\bcis .* compliance\b",
    r"\bcompliance with nist\b",
    r"\bcompliance with cis\b",
)


_SCHEMA_INTROSPECTION_PATTERNS = (
    r"\bwhat is the documented data type\b",
    r"\bwhat is the data type\b",
    r"\bwhat data type\b",
    r"\bdatatype\b",
    r"\bdata type of\b",
    r"\bdescribe (?:the )?(?:column|field)\b",
    r"\bwhat does (?:the )?(?:column|field)\b",
    r"\bwhat (?:columns|fields) (?:exist|are|does)\b",
    r"\blist (?:the )?(?:columns|fields)\b",
    r"\bwhat table contains\b",
    r"\bwhich table contains\b",
    r"\bwhat relationships? exist\b",
    r"\bwhat relationships? does\b",
    r"\bdescribe (?:the )?relationships?\b",
    r"\bwhat is the schema\b",
    r"\bschema documentation\b.*\b(?:correct|complete|accurate)\b",
)


_VAGUE_TEST_PATTERNS = (
    r"\baccurate\b",
    r"\binaccurate\b",
    r"\bcorrect\b",
    r"\bincorrect\b",
    r"\bup[- ]?to[- ]?date\b",
    r"\boutdated\b",
    r"\bsuspicious\b",
    r"\bunusual\b",
    r"\bsecurity threat\b",
    r"\bsecurity threats\b",
    r"\bsecurity risk\b",
    r"\bsecurity risks\b",
    r"\bvulnerabilit(?:y|ies)\b",
    r"\bappropriate\b",
    r"\binappropriate\b",
    r"\badequate\b",
    r"\binadequate\b",
    r"\bimproper\b",
    r"\bconcerning\b",
    r"\banomal(?:y|ies|ous)\b",
    r"\bdiscrepanc(?:y|ies)\b",
    r"\bexpected behavior\b",
    r"\bconsistent with\b",
    r"\binconsistent with\b",
    r"\binvalid\b",
)

_CONCRETE_TEST_SIGNALS = (
    "missing",
    "null",
    "blank",
    "duplicate",
    "does not match",
    "do not match",
    "doesn't match",
    "not match",
    "cannot be matched",
    "cannot be related",
    "does not correspond",
    "do not correspond",
    "no corresponding",
    "without a corresponding",
    "references an employee",
    "references a room",
    "reference does not",
    "status was",
    "status is",
    "access_result",
    "access result",
    "denied",
    "granted",
    "valid_from",
    "valid_until",
    "before",
    "after",
    "at the time",
    "at event time",
    "returned_at",
    "returned at",
    "checked_out_at",
    "checked out",
    "primary_employee_id",
    "secondary_employee_id",
    "employee_id",
    "room_id",
    "key_id",
    "camera_id",
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = _clean_text(text).lower()
    return any(term in lowered for term in terms)


def is_investigation_query(query: str) -> bool:
    cleaned = _clean_text(query)
    if not cleaned:
        return False

    if _contains_any(cleaned, _INVESTIGATION_TERMS):
        return True

    analytical = _contains_any(cleaned, _ANALYTICAL_TERMS)
    data_scope = _contains_any(cleaned, _DATA_SCOPE_TERMS)
    policy_scope = _contains_any(cleaned, _POLICY_TERMS)

    return bool(analytical and (data_scope or policy_scope))


def _is_broad_review_query(query: str) -> bool:
    """Return True when the request asks for a broad/multi-area review."""
    cleaned = _clean_text(query).lower()

    broad_terms = (
        "review the tables",
        "review tables",
        "review the records",
        "review company records",
        "review the company records",
        "security review",
        "possible security issues",
        "potential security issues",
        "security concerns",
        "possible security concerns",
        "potential security concerns",
        "possible security risks",
        "potential security risks",
        "identify anomalies",
        "find anomalies",
        "review for inconsistencies",
        "assess security",
        "evaluate security",
    )

    return any(term in cleaned for term in broad_terms)


def _contains_sql_like_text(text: str) -> bool:
    cleaned = _clean_text(text)
    return any(
        re.search(pattern, cleaned, flags=re.IGNORECASE) is not None
        for pattern in _SQL_LIKE_PATTERNS
    )


def _contains_sql_like_output(text: str) -> bool:
    cleaned = _clean_text(text)
    return any(
        re.search(pattern, cleaned, flags=re.IGNORECASE) is not None
        for pattern in _SQL_OUTPUT_PATTERNS
    )


def _contains_schema_introspection(text: str) -> bool:
    """Reject tasks that inspect schema metadata instead of operational data."""
    cleaned = _clean_text(text)

    if not cleaned:
        return False

    return any(
        re.search(
            pattern,
            cleaned,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in _SCHEMA_INTROSPECTION_PATTERNS
    )


def _contains_meta_task(text: str) -> bool:
    """Reject descriptive/schema-meta/compliance-meta investigation tasks."""
    cleaned = _clean_text(text)

    if not cleaned:
        return False

    return any(
        re.search(
            pattern,
            cleaned,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in _META_TASK_PATTERNS
    )


def _contains_vague_test_language(text: str) -> bool:
    """Return True when a task relies on a qualitative/vague judgment."""
    cleaned = _clean_text(text)

    if not cleaned:
        return False

    return any(
        re.search(
            pattern,
            cleaned,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in _VAGUE_TEST_PATTERNS
    )


def _contains_concrete_test_signal(text: str) -> bool:
    """
    Return True when the task states an observable condition SQLPlanner can
    reasonably encode from documented records.

    This is intentionally conservative; schema validation still runs
    independently after this gate.
    """
    cleaned = _clean_text(text).lower()

    if not cleaned:
        return False

    if any(signal in cleaned for signal in _CONCRETE_TEST_SIGNALS):
        return True

    # Explicit comparisons phrased in natural language.
    comparison_patterns = (
        r"\b(?:equal|equals|different from|not equal to)\b",
        r"\b(?:before|after)\s+(?:the\s+)?(?:event|timestamp|date|time)\b",
        r"\b(?:missing|duplicate|unmatched|orphaned)\s+\w+",
    )

    return any(
        re.search(
            pattern,
            cleaned,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in comparison_patterns
    )


_EXPLICIT_TEST_CONDITION_PATTERNS = (
    r"\b(?:is|are)\s+(?:not\s+)?null\b",
    r"\b(?:is|are)\s+(?:missing|blank)\b",
    r"\bduplicate(?:d|s| values?)?\b",
    r"\bdoes not match\b",
    r"\bdo not match\b",
    r"\bnot match(?:ing)?\b",
    r"\bno corresponding\b",
    r"\bwithout a corresponding\b",
    r"\bmissing (?:a |an )?(?:related|corresponding|matching)\b",
    r"\b(?:equal|equals|equal to|not equal to)\b",
    r"\b(?:before|after)\s+(?:the\s+)?(?:event|timestamp|date|time)\b",
    r"\bat (?:the )?(?:event )?time\b",
    r"\bstatus (?:is|was|equals?)\b",
    r"\baccess result (?:is|equals?|equal to)\b",
)


def _contains_explicit_test_condition(
    text: str,
) -> bool:
    """
    Return True only when the task states an observable predicate.

    Technical field names alone do not make a qualitative judgment testable.
    """
    cleaned = _clean_text(text)

    if not cleaned:
        return False

    return any(
        re.search(
            pattern,
            cleaned,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in _EXPLICIT_TEST_CONDITION_PATTERNS
    )


def _task_testability_errors(
    task: InvestigationTask,
) -> list[str]:
    """
    Keep Python out of semantic investigation design.

    An investigation task may gather relevant operational evidence even when
    the final security judgment cannot itself be encoded as a database
    predicate. The downstream LLM evaluates the combined evidence.

    Python rejects only an empty task here. SQL/pseudo-SQL, meta, schema,
    SchemaGraph, and identifier checks remain enforced independently.
    """
    question = _clean_text(task.question)

    if not question:
        return ["Task question is empty."]

    return []


def _plan_coverage_limitations(
    *,
    query: str,
    tasks: list[InvestigationTask],
    schema_graph: SchemaGraph,
) -> list[str]:
    """
    Flag plans that are too narrow for a broad review.

    Coverage is based on the deterministic SchemaGraph rather than a hard-coded
    list of security tables.
    """
    if not _is_broad_review_query(query):
        return []

    graph_table_count = len(schema_graph.table_names())

    if graph_table_count < 4:
        return []

    covered_tables = {
        table
        for task in tasks
        for table in task.required_tables
        if schema_graph.has_table(table)
    }

    cross_table_tasks = sum(
        1
        for task in tasks
        if len(task.required_tables) > 1 and bool(task.relationship_paths)
    )

    limitations: list[str] = []

    if len(tasks) < 3:
        limitations.append(
            (
                "Broad review produced fewer than three usable investigation "
                "tasks even though the SchemaGraph contains multiple "
                "operational data domains."
            )
        )

    if len(covered_tables) < 3:
        limitations.append(
            (
                "Broad review is too narrowly scoped: usable tasks cover only "
                f"{len(covered_tables)} documented table(s) out of "
                f"{graph_table_count}. Additional distinct schema-supported "
                "operational areas should be considered."
            )
        )

    if cross_table_tasks == 0:
        limitations.append(
            (
                "Broad review contains no validated cross-table investigation "
                "even though SchemaGraph documents multiple relationships. "
                "At least one concrete relational consistency check should be "
                "included when supported by the user request and schema."
            )
        )

    return list(dict.fromkeys(limitations))


_TABLE_SCOPE_GENERIC_TOKENS = {
    "access",
    "log",
    "record",
    "records",
    "event",
    "events",
    "data",
    "table",
    "tables",
    "history",
    "summary",
}


_RELATIONAL_INTENT_PATTERNS = (
    r"\bdoes not match\b",
    r"\bdo not match\b",
    r"\bnot match(?:ing)?\b",
    r"\bno corresponding\b",
    r"\bwithout a corresponding\b",
    r"\breference(?:s|d)?\b",
    r"\brelated record\b",
    r"\bauthorization rule\b",
    r"\bauthorization rules\b",
    r"\bcorrelat(?:e|ed|ion)\b",
    r"\bcompared? with\b",
    r"\bstatus at\b",
    r"\bat (?:the )?event time\b",
    r"\bresponsible employee\b",
    r"\bmissing (?:a |an )?(?:related|matching|corresponding)\b",
)


def _normalize_scope_token(
    token: str,
) -> str:
    cleaned = re.sub(
        r"[^a-z0-9]+",
        "",
        _clean_text(token).lower(),
    )

    if cleaned.endswith("ies") and len(cleaned) > 4:
        return cleaned[:-3] + "y"

    if cleaned.endswith("s") and len(cleaned) > 4:
        return cleaned[:-1]

    return cleaned


def _scope_tokens(
    text: str,
) -> set[str]:
    tokens = {
        _normalize_scope_token(token)
        for token in re.findall(
            r"[A-Za-z][A-Za-z0-9]*",
            _clean_text(text),
        )
    }

    return {
        token for token in tokens if token and token not in _TABLE_SCOPE_GENERIC_TOKENS
    }


def _table_scope_tokens(
    table_name: str,
) -> set[str]:
    return {
        token
        for token in (
            _normalize_scope_token(part) for part in _clean_text(table_name).split("_")
        )
        if token and token not in _TABLE_SCOPE_GENERIC_TOKENS
    }


def _task_has_relational_intent(
    task: InvestigationTask,
) -> bool:
    task_text = " ".join(
        [
            task.title,
            task.question,
            task.rationale,
        ]
    )

    return any(
        re.search(
            pattern,
            task_text,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in _RELATIONAL_INTENT_PATTERNS
    )


def _column_owner_tables(
    identifier: str,
    schema_graph: SchemaGraph,
) -> list[str]:
    cleaned = _clean_text(identifier).lower()

    return [
        table_name
        for table_name, table in schema_graph.tables.items()
        if table.has_column(cleaned)
    ]


def _complete_required_tables_from_task(
    task: InvestigationTask,
    schema_graph: SchemaGraph,
) -> tuple[list[str], list[str]]:
    """
    Normalize the LLM-selected required-table contract without trying to infer
    semantic table endpoints from natural-language task wording.

    Python is intentionally limited here:
    - keep tables explicitly selected by the LLM;
    - add table names explicitly written in task prose;
    - add a table only when a technical identifier has exactly one documented
      owner in SchemaGraph.

    Python does NOT guess that words such as "employee", "status", "room",
    "clock", or "access" imply an additional table. If the LLM omitted a needed
    evidence domain, the ledger/Agent 3 supplemental loop will identify the gap
    after valid evidence is executed.
    """
    selected = list(
        dict.fromkeys(
            _clean_text(table).lower()
            for table in task.required_tables
            if _clean_text(table)
        )
    )

    # Explicit documented table names in task prose are safe to preserve.
    for table in _graph_tables_mentioned(
        task,
        schema_graph,
    ):
        if table not in selected:
            selected.append(table)

    task_text = " ".join(
        [
            task.title,
            task.question,
            task.rationale,
            *task.requested_outputs,
        ]
    )

    # A technical identifier with exactly one documented owner may safely add
    # that owner. Ambiguous identifiers are left to the LLM rather than guessed.
    for identifier in sorted(_technical_identifiers(task_text)):
        owners = _column_owner_tables(
            identifier,
            schema_graph,
        )

        if len(owners) == 1 and owners[0] not in selected:
            selected.append(owners[0])

    return (
        list(dict.fromkeys(selected)),
        [],
    )


def _graph_tables_mentioned(
    task: InvestigationTask,
    schema_graph: SchemaGraph,
) -> list[str]:
    task_text = " ".join(
        [
            task.title,
            task.question,
            task.rationale,
            *task.requested_outputs,
        ]
    ).lower()

    return [
        table
        for table in schema_graph.table_names()
        if re.search(
            rf"\b{re.escape(table)}\b",
            task_text,
            flags=re.IGNORECASE,
        )
    ]


def _resolve_required_tables_and_paths(
    task: InvestigationTask,
    schema_graph: SchemaGraph,
    *,
    max_hops: int = 4,
) -> tuple[bool, list[str]]:
    errors: list[str] = []

    selected_tables = list(
        dict.fromkeys(
            _clean_text(table).lower()
            for table in task.required_tables
            if _clean_text(table)
        )
    )

    if not selected_tables:
        selected_tables = _graph_tables_mentioned(
            task,
            schema_graph,
        )

    undocumented = [
        table for table in selected_tables if not schema_graph.has_table(table)
    ]

    if undocumented:
        return (
            False,
            ["Undocumented required table(s): " + ", ".join(sorted(undocumented))],
        )

    if len(selected_tables) <= 1:
        task.required_tables = selected_tables
        task.relationship_paths = []
        task.required_schema_documents = [
            Path(schema_graph.tables[table].source_document).stem
            for table in selected_tables
            if table in schema_graph.tables
        ]
        return (True, [])

    connected: set[str] = {selected_tables[0]}
    unresolved: set[str] = set(selected_tables[1:])
    resolved_paths: list[RelationshipPath] = []

    while unresolved:
        best_path: RelationshipPath | None = None
        best_target: str | None = None

        for source_table in sorted(connected):
            for target_table in sorted(unresolved):
                candidate = schema_graph.shortest_planning_path(
                    source_table,
                    target_table,
                    max_hops=max_hops,
                )

                if candidate is None:
                    continue

                if (
                    best_path is None
                    or candidate.hop_count < best_path.hop_count
                    or (
                        candidate.hop_count == best_path.hop_count
                        and candidate.tables < best_path.tables
                    )
                ):
                    best_path = candidate
                    best_target = target_table

        if best_path is None or best_target is None:
            return (
                False,
                [
                    "No documented SchemaGraph planning path connects remaining "
                    "table(s): " + ", ".join(sorted(unresolved))
                ],
            )

        resolved_paths.append(best_path)
        connected.update(best_path.tables)
        unresolved.remove(best_target)

    expanded_tables: list[str] = []

    for table in selected_tables:
        if table not in expanded_tables:
            expanded_tables.append(table)

    for path in resolved_paths:
        for table in path.tables:
            if table not in expanded_tables:
                expanded_tables.append(table)

    task.required_tables = expanded_tables
    task.relationship_paths = list(
        dict.fromkeys(path.render() for path in resolved_paths)
    )
    task.required_schema_documents = list(
        dict.fromkeys(
            Path(schema_graph.tables[table].source_document).stem
            for table in expanded_tables
            if table in schema_graph.tables
        )
    )

    return (True, [])


def _schema_document_names(schema_context: str) -> set[str]:
    names = {
        match.group(1).lower()
        for match in re.finditer(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\.md\b",
            schema_context,
            flags=re.IGNORECASE,
        )
    }

    if names:
        return names

    return {
        token.lower()
        for token in re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            schema_context,
        )
        if "_" in token
    }


def _schema_identifier_tokens(schema_context: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            schema_context,
        )
    }


def _technical_identifiers(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            _clean_text(text),
        )
        if "_" in token
    }


def _task_schema_is_supported(
    task: InvestigationTask,
    *,
    schema_documents: set[str],
    schema_identifiers: set[str],
) -> tuple[bool, list[str]]:
    """
    Validate only planner-level schema references.

    InvestigationPlanner operates at the SEMANTIC evidence layer. Therefore:
    - required schema documents must be documented;
    - required tables are validated separately by SchemaGraph;
    - natural-language task text and requested_outputs are NOT validated as
      physical column identifiers.

    A semantic label such as "employee_name", "clock_state", or
    "access_event_time" must not be rejected merely because no physical column
    has that exact name. SQLPlanner is responsible for mapping semantic evidence
    concepts to documented physical fields, and downstream SQL validation is the
    hard authority for actual table/column correctness.

    schema_identifiers is retained in the signature for compatibility with the
    existing validation pipeline, but is intentionally unused here.
    """
    del schema_identifiers

    errors: list[str] = []

    normalized_documents = [
        _clean_text(document).removesuffix(".md").lower()
        for document in task.required_schema_documents
        if _clean_text(document)
    ]

    for document in normalized_documents:
        if schema_documents and document not in schema_documents:
            errors.append(f"Undocumented schema document: {document}")

    return (not errors, errors)


def _normalize_requested_outputs(outputs: list[str]) -> list[str]:
    normalized: list[str] = []

    for output in outputs:
        cleaned = _clean_text(output)

        if not cleaned:
            continue

        if _contains_sql_like_output(cleaned):
            continue

        if cleaned not in normalized:
            normalized.append(cleaned)

    return normalized


def _extract_structured_response(response: Any) -> InvestigationPlan:
    raw = getattr(response, "raw", None)

    if isinstance(raw, InvestigationPlan):
        return raw

    if isinstance(raw, dict):
        return InvestigationPlan.model_validate(raw)

    if isinstance(response, InvestigationPlan):
        return response

    parsed = (getattr(response, "additional_kwargs", {}) or {}).get("parsed")

    if isinstance(parsed, InvestigationPlan):
        return parsed

    if isinstance(parsed, dict):
        return InvestigationPlan.model_validate(parsed)

    text = _clean_text(getattr(response, "text", response))

    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    if not text:
        raise ValueError("The investigation planner returned an empty response.")

    return InvestigationPlan.model_validate_json(text)


_INVESTIGATION_REPAIR_PROMPT = """
Your previous investigation plan produced no usable tasks after deterministic
validation.

USER REQUEST
------------
{query}

AUTHORITATIVE SCHEMA
--------------------
{schema_context}

SCHEMAGRAPH TABLE CATALOG
-------------------------
{graph_table_catalog}

SCHEMAGRAPH DOCUMENTED EVIDENCE-PLANNING RELATIONSHIPS
-------------------------------------------
{graph_relationship_catalog}

Use the planning graph to choose relevant required_tables. Planning paths show documented evidence-domain connectivity only. Do not invent join columns or SQL paths. Leave relationship_paths empty.

RETRIEVED POLICY / STANDARDS EVIDENCE
-------------------------------------
{policy_context}

REJECTION / LIMITATION REASONS
------------------------------
{rejection_reasons}

Create one corrected InvestigationPlan.

CORRECTION RULES
----------------
1. Return only natural-language operational evidence questions.
2. Do not write SQL or pseudo-SQL. Normal English words such as "where",
   "from", "before", "after", and "in" are allowed and should not be avoided.
3. Do not ask what a table means, what relationships exist, whether a dataset
   is sufficient, or whether a table/database proves NIST or CIS compliance.
4. Do not invent dates, relative time windows, thresholds, business values,
   employees, rooms, authorization rules, or relationships.
5. Use only documented table names and documented relationships when referring
   to physical database structure.

   Natural-language evidence concepts and requested_outputs are semantic labels
   and do not have to match physical column names. Do not turn a semantic label
   into an asserted database column; SQLPlanner will perform that mapping.
6. Describe WHAT records or evidence should be checked, not HOW to query them.
7. Prefer simple questions about documented events, statuses, results,
   timestamps, references, or inconsistencies.

8. Every repaired task MUST contain a concrete observable test condition.
   Do not ask whether data is accurate, suspicious, unusual, adequate,
   appropriate, risky, vulnerable, or otherwise concerning unless the same
   question explicitly states the documented record condition that defines it.

9. Examples of acceptable conditions include:
   - missing or unmatched documented references;
   - null/missing documented fields;
   - duplicate documented identifiers when duplicate detection is meaningful;
   - documented event result/status values;
   - an event inconsistent with a documented authorization rule;
   - status at event time when supported by documented temporal relationships.

10. If you cannot state the concrete test condition using documented schema,
    omit the task and record the missing rule/limitation instead.

11. For a broad multi-source review, produce several distinct non-overlapping
    tasks when documented relationships support them. Do not reduce the repair
    to one generic compliance question if multiple concrete checks are possible.

12. Prefer cross-table or cross-record checks when documented relationships
    make those checks meaningful.

13. Produce at least one task if the schema supports at least one concrete
    security-relevant factual analysis.

14. Reject schema-introspection ideas such as asking for data types,
    column definitions, field descriptions, table contents, or schema
    relationships. Those facts are already supplied.

15. Use SCHEMAGRAPH to select concrete operational tables for comparison.
    Populate required_tables only with documented table names, and include
    EVERY endpoint table required to establish the stated condition or return
    the requested evidence. Do not submit a relational question with only its
    event/source table.

    Leave relationship_paths empty so Python resolves the legal path.

16. A repaired task must test an operational record condition. Merely listing
    a column's values or asking about its metadata is not sufficient.

17. Limit the corrected plan to the most useful {max_tasks} tasks.

Before returning a TARGETED correlation repair, ensure every factual side/domain
in the user's comparison is represented by the repaired tasks. If necessary,
create complementary tasks with shared correlation fields.
"""


def _merge_validated_investigation_plans(
    *,
    initial: InvestigationPlan,
    repaired: InvestigationPlan,
) -> InvestigationPlan:
    """
    Merge two plans that have ALREADY passed task-level validation.

    Do not run the merged plan through task validation again. Validation mutates
    task metadata (for example, required_tables / relationship_paths), so a
    second validation pass can incorrectly reject tasks that were valid in the
    individual plans.

    The merge:
    - preserves already validated tasks;
    - de-duplicates semantically identical task questions;
    - assigns unique task IDs;
    - preserves limitations from both plans;
    - caps the final plan at MAX_INVESTIGATION_TASKS.
    """
    merged_tasks: list[InvestigationTask] = []
    seen_questions: set[str] = set()
    seen_ids: set[str] = set()

    for source_task in [
        *initial.tasks,
        *repaired.tasks,
    ]:
        question_key = re.sub(
            r"\s+",
            " ",
            _clean_text(source_task.question).lower(),
        )

        if not question_key or question_key in seen_questions:
            continue

        if len(merged_tasks) >= MAX_INVESTIGATION_TASKS:
            break

        seen_questions.add(question_key)

        base_id = _clean_text(source_task.task_id) or (f"T{len(merged_tasks) + 1:03d}")
        task_id = base_id
        suffix = 2

        while task_id in seen_ids:
            task_id = f"{base_id}_{suffix}"
            suffix += 1

        seen_ids.add(task_id)

        # Pydantic copy keeps the individually validated task content while
        # allowing a unique merged task ID.
        merged_tasks.append(
            source_task.model_copy(
                update={
                    "task_id": task_id,
                }
            )
        )

    return InvestigationPlan(
        purpose=initial.purpose or repaired.purpose,
        is_investigation=True,
        tasks=merged_tasks,
        assumptions=list(
            dict.fromkeys(
                [
                    *initial.assumptions,
                    *repaired.assumptions,
                ]
            )
        ),
        missing_business_rules=list(
            dict.fromkeys(
                [
                    *initial.missing_business_rules,
                    *repaired.missing_business_rules,
                ]
            )
        ),
    )


class InvestigationPlanner:
    """Create a focused, evidence-grounded investigation plan with up to MAX_INVESTIGATION_TASKS validated tasks."""

    def __init__(
        self,
        llm: LLM,
        schema_graph: SchemaGraph | None = None,
        schema_directory: str | Path | None = None,
    ) -> None:
        self._structured_llm = llm.as_structured_llm(InvestigationPlan)
        self._initial_plan_review_llm = llm.as_structured_llm(SupplementalPlanReview)
        self._supplemental_review_llm = llm.as_structured_llm(SupplementalPlanReview)

        if schema_graph is not None:
            self._schema_graph = schema_graph
        else:
            if schema_directory is None:
                project_root = Path(__file__).resolve().parents[1]
                schema_directory = project_root / "docs" / "schema"

            self._schema_graph = SchemaGraph.from_directory(schema_directory)

        print("\n[InvestigationPlanner] ⚙  " "Initialising investigation planner…")
        print("[InvestigationPlanner]    Output : " "natural-language evidence tasks")
        print(
            "[InvestigationPlanner]    Inputs : "
            "query + raw schema + SchemaGraph + policy evidence"
        )
        print("[InvestigationPlanner]    SQL    : prohibited")
        print(
            "[InvestigationPlanner]    Graph  : "
            f"{len(self._schema_graph.tables)} tables / "
            f"{len(self._schema_graph.planning_relationships)} planning relationships / "
            f"{len(self._schema_graph.relationships)} SQL relationships"
        )
        print(
            "[InvestigationPlanner]    Guard  : "
            "SQL/pseudo-SQL, meta/schema-introspection, undocumented tables, "
            "and invalid SchemaGraph planning paths are rejected"
        )
        print(
            "[InvestigationPlanner]    Semantics: "
            "task questions/outputs are evidence concepts; SQLPlanner maps "
            "them to physical columns"
        )
        print(
            "[InvestigationPlanner]    Repair : "
            "targeted initial plans use up to "
            f"{MAX_INITIAL_TARGETED_PLAN_ATTEMPTS} generate/review cycles; "
            "broad initial plans retain bounded repair; supplemental plans use "
            f"up to {MAX_SUPPLEMENTAL_PLAN_ATTEMPTS} generate/review cycles"
        )
        print("[InvestigationPlanner]    Limit  : " f"{MAX_INVESTIGATION_TASKS} tasks")

    def _validate_and_normalize_plan(
        self,
        result: InvestigationPlan,
        *,
        cleaned_query: str,
        cleaned_schema: str,
    ) -> InvestigationPlan:
        """
        Apply planner-level deterministic guards to one generated plan.

        Hard checks here are limited to structural/safety concerns appropriate
        to the semantic planning layer: SQL-like task text, meta/schema
        introspection, documented tables, and SchemaGraph relationships.

        Semantic task labels and requested_outputs are intentionally NOT treated
        as physical column identifiers. SQLPlanner and downstream SQL validation
        own physical schema correctness.
        """
        result.is_investigation = True
        result.tasks = result.tasks[:MAX_INVESTIGATION_TASKS]

        schema_documents = _schema_document_names(cleaned_schema)
        schema_identifiers = _schema_identifier_tokens(cleaned_schema)

        seen_ids: set[str] = set()
        seen_questions: set[str] = set()
        unique_tasks: list[InvestigationTask] = []
        rejected_reasons: list[str] = []

        for index, task in enumerate(
            result.tasks,
            start=1,
        ):
            task_id = _clean_text(task.task_id) or f"task_{index}"

            if task_id in seen_ids:
                task_id = f"{task_id}_{index}"

            seen_ids.add(task_id)
            task.task_id = task_id

            task.title = _clean_text(task.title) or f"Investigation task {index}"
            task.question = _clean_text(task.question)
            task.rationale = _clean_text(task.rationale)

            if not task.question:
                rejected_reasons.append(f"{task_id}: empty task question.")
                continue

            if _contains_sql_like_text(task.question):
                rejected_reasons.append(
                    f"{task_id}: task question contained SQL "
                    "or pseudo-SQL and was rejected."
                )
                continue

            if _contains_schema_introspection(task.question):
                rejected_reasons.append(
                    (
                        f"{task_id}: task asked for schema metadata or "
                        "schema introspection rather than operational evidence."
                    )
                )
                continue

            if _contains_meta_task(task.question):
                rejected_reasons.append(
                    (
                        f"{task_id}: task was descriptive, "
                        "schema-oriented, or asked the database "
                        "layer to determine policy compliance."
                    )
                )
                continue

            testability_errors = _task_testability_errors(task)

            if testability_errors:
                rejected_reasons.append(
                    (
                        f"{task_id}: testability validation failed: "
                        + "; ".join(testability_errors)
                    )
                )
                continue

            task.required_schema_documents = list(
                dict.fromkeys(
                    _clean_text(document).removesuffix(".md")
                    for document in task.required_schema_documents
                    if _clean_text(document)
                )
            )

            task.requested_outputs = _normalize_requested_outputs(
                task.requested_outputs
            )

            task.relationship_paths = []

            (
                completed_tables,
                scope_errors,
            ) = _complete_required_tables_from_task(
                task,
                self._schema_graph,
            )

            task.required_tables = completed_tables

            if scope_errors:
                rejected_reasons.append(
                    f"{task_id}: table-scope validation failed: "
                    + "; ".join(scope_errors)
                )
                continue

            (
                graph_supported,
                graph_errors,
            ) = _resolve_required_tables_and_paths(
                task,
                self._schema_graph,
            )

            if not graph_supported:
                rejected_reasons.append(
                    f"{task_id}: SchemaGraph planning validation failed: "
                    + "; ".join(graph_errors)
                )
                continue

            (
                schema_supported,
                schema_errors,
            ) = _task_schema_is_supported(
                task,
                schema_documents=schema_documents,
                schema_identifiers=schema_identifiers,
            )

            if not schema_supported:
                rejected_reasons.append(
                    f"{task_id}: schema validation failed: " + "; ".join(schema_errors)
                )
                continue

            question_key = re.sub(
                r"\s+",
                " ",
                task.question.lower(),
            )

            if question_key in seen_questions:
                continue

            seen_questions.add(question_key)

            print(
                "[InvestigationPlanner]    Accepted task "
                f"{task.task_id} tables: {task.required_tables}"
            )

            if task.relationship_paths:
                print(
                    "[InvestigationPlanner]    Resolved path(s) for "
                    f"{task.task_id}: {task.relationship_paths}"
                )

            unique_tasks.append(task)

        result.tasks = unique_tasks

        result.assumptions = list(
            dict.fromkeys(
                _clean_text(value) for value in result.assumptions if _clean_text(value)
            )
        )

        result.missing_business_rules = list(
            dict.fromkeys(
                _clean_text(value)
                for value in result.missing_business_rules
                if _clean_text(value)
            )
        )

        for reason in rejected_reasons:
            if reason not in result.missing_business_rules:
                result.missing_business_rules.append(reason)

        if not result.tasks and not result.missing_business_rules:
            result.missing_business_rules.append(
                (
                    "The planner could not identify a "
                    "schema-supported natural-language "
                    "investigation task from the supplied "
                    "request and evidence."
                )
            )

        coverage_limitations = _plan_coverage_limitations(
            query=cleaned_query,
            tasks=result.tasks,
            schema_graph=self._schema_graph,
        )

        for limitation in coverage_limitations:
            if limitation not in result.missing_business_rules:
                result.missing_business_rules.append(limitation)

        return result

    def _generate_plan(
        self,
        *,
        prompt: str,
    ) -> InvestigationPlan:
        response = _complete_with_system(
            self._structured_llm,
            system_prompt=INVESTIGATION_PLANNER_SYSTEM_PROMPT,
            user_prompt=prompt,
        )

        return _extract_structured_response(response)

    def _review_initial_targeted_plan(
        self,
        *,
        plan: InvestigationPlan,
        query: str,
        ledger_context: str,
        graph_table_catalog: str,
        graph_relationship_catalog: str,
    ) -> SupplementalPlanReview:
        """Review whole-plan evidence coverage against the persistent ledger."""
        prompt = _INITIAL_TARGETED_PLAN_REVIEW_PROMPT.format(
            query=query,
            ledger_context=(
                _clean_text(ledger_context)
                or "(no persistent ledger requirements were supplied)"
            ),
            proposed_plan=plan.model_dump_json(indent=2),
            graph_table_catalog=graph_table_catalog,
            graph_relationship_catalog=graph_relationship_catalog,
        )

        try:
            response = _complete_with_system(
                self._initial_plan_review_llm,
                system_prompt=INVESTIGATION_PLANNER_SYSTEM_PROMPT,
                user_prompt=prompt,
            )
            raw = getattr(response, "raw", None)

            if isinstance(raw, SupplementalPlanReview):
                return raw

            if isinstance(raw, dict):
                return SupplementalPlanReview.model_validate(raw)

            parsed = (getattr(response, "additional_kwargs", {}) or {}).get("parsed")

            if isinstance(parsed, SupplementalPlanReview):
                return parsed

            if isinstance(parsed, dict):
                return SupplementalPlanReview.model_validate(parsed)

            text = _clean_text(getattr(response, "text", response))
            text = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()

            return SupplementalPlanReview.model_validate_json(text)

        except Exception as exc:
            return SupplementalPlanReview(
                sufficient=False,
                reason=(
                    "Initial targeted-plan self-review could not be completed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                missing_requirements=[
                    "A reliable review of cross-domain evidence coverage is required."
                ],
            )

    def plan(
        self,
        query: str,
        schema_context: str,
        policy_context: str = "",
        *,
        force_investigation: bool = False,
        ledger_context: str = "",
    ) -> InvestigationPlan:
        """
        Create one broad or targeted investigation decomposition.

        When force_investigation=True, Agent 1's evidence contract is
        authoritative and this planner must attempt task generation even when
        its backward-compatible local query heuristic would not independently
        classify the wording as an investigation.

        One bounded repair may be attempted after deterministic validation.
        """
        cleaned_query = _clean_text(query)
        cleaned_schema = _clean_text(schema_context)
        cleaned_policy = _clean_text(policy_context)

        graph_table_catalog = self._schema_graph.render_table_catalog()
        graph_relationship_catalog = (
            self._schema_graph.render_planning_relationship_catalog()
        )

        locally_detected_investigation = is_investigation_query(cleaned_query)

        if not locally_detected_investigation and not force_investigation:
            return InvestigationPlan(
                purpose=cleaned_query,
                is_investigation=False,
                tasks=[],
            )

        if force_investigation and not locally_detected_investigation:
            print(
                "[InvestigationPlanner]    Upstream contract override: "
                "Agent 1 requires investigation; local text heuristic bypassed."
            )

        if not cleaned_schema:
            return InvestigationPlan(
                purpose=cleaned_query,
                is_investigation=True,
                tasks=[],
                missing_business_rules=[
                    "No authoritative schema context " "was supplied."
                ],
            )

        investigation_shape = (
            "broad" if _is_broad_review_query(cleaned_query) else "targeted"
        )

        print(
            "[InvestigationPlanner]    Investigation shape: " f"{investigation_shape}"
        )

        if investigation_shape == "targeted":
            print(
                "[InvestigationPlanner]    Targeted rule: preserve every "
                "comparison domain across complementary evidence tasks."
            )

        policy_for_prompt = (
            cleaned_policy
            if cleaned_policy
            else (
                "No policy evidence was supplied. "
                "Plan only analyses supported by the schema "
                "and user request."
            )
        )

        initial_prompt = _INVESTIGATION_PROMPT.format(
            query=cleaned_query,
            ledger_context=_clean_text(ledger_context) or "(no ledger entries yet)",
            schema_context=cleaned_schema,
            graph_table_catalog=graph_table_catalog,
            graph_relationship_catalog=graph_relationship_catalog,
            policy_context=policy_for_prompt,
            max_tasks=MAX_INVESTIGATION_TASKS,
        )

        print(
            "[InvestigationPlanner]    SchemaGraph context: "
            f"{len(graph_table_catalog)} table chars / "
            f"{len(graph_relationship_catalog)} relationship chars"
        )

        # TARGETED investigations use bounded generate -> validate -> self-review.
        #
        # The LLM proposes evidence tasks and Python validates schema/safety.
        # If the LLM's whole-plan review says material evidence coverage is still
        # missing, that review is fed into another bounded planning cycle before
        # execution. Python does not decide which evidence domain should be added.
        if investigation_shape == "targeted":
            feedback: list[str] = []
            last_targeted_plan = InvestigationPlan(
                purpose=cleaned_query,
                is_investigation=True,
                tasks=[],
            )

            for attempt in range(1, MAX_INITIAL_TARGETED_PLAN_ATTEMPTS + 1):
                attempt_prompt = initial_prompt

                if feedback:
                    attempt_prompt += (
                        "\n\nPREVIOUS TARGETED PLAN FEEDBACK\n"
                        "---------------------------------\n"
                        + "\n".join(f"- {item}" for item in feedback)
                        + "\n\nGenerate one or more useful factual evidence "
                        "tasks. Prefer tasks that can execute now. Do not wait "
                        "for one task to solve the entire investigation."
                    )

                print(
                    "[InvestigationPlanner]    Initial targeted reasoning cycle "
                    f"{attempt}/{MAX_INITIAL_TARGETED_PLAN_ATTEMPTS}"
                )

                generated = self._generate_plan(prompt=attempt_prompt)

                validated = self._validate_and_normalize_plan(
                    generated,
                    cleaned_query=cleaned_query,
                    cleaned_schema=cleaned_schema,
                )

                last_targeted_plan = validated

                if not validated.tasks:
                    reasons = validated.missing_business_rules or [
                        "No executable targeted tasks survived validation."
                    ]
                    feedback = [
                        "No executable task survived schema/safety validation: "
                        + "; ".join(reasons)
                    ]
                    print(
                        "[InvestigationPlanner]    No executable targeted task "
                        "yet; giving the LLM another reasoning cycle."
                    )
                    continue

                # Advisory coverage reflection. It may identify what remains
                # missing, but it no longer vetoes already-valid evidence tasks.
                review = self._review_initial_targeted_plan(
                    plan=validated,
                    query=cleaned_query,
                    ledger_context=_clean_text(ledger_context),
                    graph_table_catalog=graph_table_catalog,
                    graph_relationship_catalog=graph_relationship_catalog,
                )

                print(
                    "[InvestigationPlanner]    Initial targeted self-review: "
                    f"{'COMPLETE' if review.sufficient else 'PARTIAL'} — "
                    f"{review.reason}"
                )

                if not review.sufficient:
                    coverage_note = "Targeted plan coverage gap: " + review.reason
                    if coverage_note not in validated.missing_business_rules:
                        validated.missing_business_rules.append(coverage_note)

                    feedback = [
                        "The previous targeted plan was judged incomplete.",
                        "Review reason: " + review.reason,
                    ]

                    if review.missing_requirements:
                        feedback.append(
                            "Still required: " + "; ".join(review.missing_requirements)
                        )

                    if attempt < MAX_INITIAL_TARGETED_PLAN_ATTEMPTS:
                        print(
                            "[InvestigationPlanner]    Initial targeted plan was "
                            "PARTIAL; using another bounded LLM planning cycle "
                            "instead of executing a knowingly incomplete plan."
                        )
                        continue

                    print(
                        "[InvestigationPlanner]    Final targeted plan remains "
                        "PARTIAL after the bounded retry budget; returning the "
                        "best validated plan with explicit coverage limitations."
                    )
                else:
                    print(
                        "[InvestigationPlanner]    Targeted plan self-review: "
                        "SUFFICIENT."
                    )

                print(
                    "[InvestigationPlanner]    Returning "
                    f"{len(validated.tasks)} targeted task(s)."
                )
                return validated

            limitations = list(last_targeted_plan.missing_business_rules)
            limitations.append(
                "The targeted planner exhausted "
                f"{MAX_INITIAL_TARGETED_PLAN_ATTEMPTS} reasoning cycles "
                "without producing any schema-valid executable task."
            )
            last_targeted_plan.missing_business_rules = list(dict.fromkeys(limitations))

            print(
                "[InvestigationPlanner]    Targeted planning stopped because no "
                "executable task could be produced."
            )
            return last_targeted_plan

        # BROAD investigations retain the prior initial generation + bounded
        # repair behavior.
        initial = self._generate_plan(prompt=initial_prompt)

        initial = self._validate_and_normalize_plan(
            initial,
            cleaned_query=cleaned_query,
            cleaned_schema=cleaned_schema,
        )

        coverage_shortfall = any(
            (
                "Broad review produced fewer than three usable investigation tasks"
                in reason
                or "Broad review is too narrowly scoped" in reason
                or "Broad review contains no validated cross-table investigation"
                in reason
            )
            for reason in initial.missing_business_rules
        )

        if initial.tasks and not coverage_shortfall:
            return initial

        initial_reasons = list(initial.missing_business_rules)

        if initial.tasks and coverage_shortfall:
            initial_reasons.append(
                (
                    "The initial plan was too narrow for a broad review. "
                    "Create additional distinct, concrete, schema-supported "
                    "tasks that cover other documented operational tables and "
                    "at least one valid cross-table relationship when the "
                    "SchemaGraph supports it. required_tables must include all "
                    "endpoint tables needed by each relational task."
                )
            )

        print(
            "[InvestigationPlanner]    Initial plan requires bounded repair: "
            f"{len(initial.tasks)} usable task(s), "
            f"coverage_shortfall={coverage_shortfall}."
        )

        for reason in initial_reasons:
            print("[InvestigationPlanner]      - " f"{reason}")

        rejection_text = (
            "\n".join(f"- {reason}" for reason in initial_reasons)
            or "- No usable tasks survived validation."
        )

        repair_prompt = _INVESTIGATION_REPAIR_PROMPT.format(
            query=cleaned_query,
            schema_context=cleaned_schema,
            graph_table_catalog=graph_table_catalog,
            graph_relationship_catalog=graph_relationship_catalog,
            policy_context=policy_for_prompt,
            rejection_reasons=rejection_text,
            max_tasks=MAX_INVESTIGATION_TASKS,
        )

        repaired = self._generate_plan(prompt=repair_prompt)

        repaired = self._validate_and_normalize_plan(
            repaired,
            cleaned_query=cleaned_query,
            cleaned_schema=cleaned_schema,
        )

        if repaired.tasks:
            if initial.tasks:
                # Both plans have already passed _validate_and_normalize_plan().
                # Merge the validated task objects directly. Re-validating the
                # merged plan can incorrectly reject already-valid tasks because
                # task normalization is intentionally mutating.
                merged = _merge_validated_investigation_plans(
                    initial=initial,
                    repaired=repaired,
                )

                print(
                    "[InvestigationPlanner]    Repair merged with initial plan: "
                    f"{len(merged.tasks)} usable task(s) "
                    f"(maximum {MAX_INVESTIGATION_TASKS})."
                )

                return merged

            print(
                "[InvestigationPlanner]    Repair succeeded: "
                f"{len(repaired.tasks)} usable task(s)."
            )
            return repaired

        print("[InvestigationPlanner]    Repair produced 0 usable tasks.")

        combined_limitations = list(
            dict.fromkeys(
                [
                    *initial_reasons,
                    *repaired.missing_business_rules,
                    (
                        "One bounded investigation-plan repair "
                        "was attempted but no usable tasks "
                        "survived validation."
                    ),
                ]
            )
        )

        repaired.missing_business_rules = combined_limitations

        return repaired

    def _review_supplemental_plan(
        self,
        *,
        plan: InvestigationPlan,
        supplemental_request: str,
        original_query: str,
        existing_evidence_summary: str,
        graph_table_catalog: str,
        ledger_context: str = "",
    ) -> SupplementalPlanReview:
        """Ask the LLM to reflect on whether the proposed plan closes the stated evidence gap."""
        prompt = _SUPPLEMENTAL_PLAN_REVIEW_PROMPT.format(
            supplemental_request=supplemental_request,
            original_query=original_query or "(not supplied)",
            existing_evidence_summary=(
                existing_evidence_summary
                or "(no successful prior evidence metadata supplied)"
            ),
            ledger_context=_clean_text(ledger_context) or "(no ledger entries yet)",
            proposed_plan=plan.model_dump_json(indent=2),
            graph_table_catalog=graph_table_catalog,
        )

        try:
            response = _complete_with_system(
                self._supplemental_review_llm,
                system_prompt=INVESTIGATION_PLANNER_SYSTEM_PROMPT,
                user_prompt=prompt,
            )
            raw = getattr(response, "raw", None)

            if isinstance(raw, SupplementalPlanReview):
                return raw

            if isinstance(raw, dict):
                return SupplementalPlanReview.model_validate(raw)

            parsed = (getattr(response, "additional_kwargs", {}) or {}).get("parsed")

            if isinstance(parsed, SupplementalPlanReview):
                return parsed

            if isinstance(parsed, dict):
                return SupplementalPlanReview.model_validate(parsed)

            text = _clean_text(getattr(response, "text", response))
            text = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()

            return SupplementalPlanReview.model_validate_json(text)

        except Exception as exc:
            return SupplementalPlanReview(
                sufficient=False,
                reason=(
                    "Supplemental plan self-review could not be completed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                missing_requirements=[
                    "A reliable self-review of evidence-gap coverage is required."
                ],
            )

    def plan_supplemental(
        self,
        *,
        original_query: str,
        supplemental_request: str,
        existing_evidence_summary: str,
        schema_context: str,
        policy_context: str = "",
        ledger_context: str = "",
    ) -> InvestigationPlan:
        """
        Plan only the concrete missing evidence identified by the current
        investigation sufficiency reviewer (Agent 2 or Agent 3).

        The LLM gets bounded generate -> validate -> self-review cycles.
        A supplemental plan that the LLM itself judges insufficient is revised
        within the bounded planning budget rather than executed knowingly.
        """
        cleaned_original = _clean_text(original_query)
        cleaned_request = _clean_text(supplemental_request)
        cleaned_existing = _clean_text(existing_evidence_summary)
        cleaned_schema = _clean_text(schema_context)
        cleaned_policy = _clean_text(policy_context)

        if not cleaned_request:
            return InvestigationPlan(
                purpose="Supplemental evidence request was empty.",
                is_investigation=True,
                tasks=[],
                missing_business_rules=["No missing-evidence objective was supplied."],
            )

        if not cleaned_schema:
            return InvestigationPlan(
                purpose=cleaned_request,
                is_investigation=True,
                tasks=[],
                missing_business_rules=[
                    "No authoritative schema context was supplied."
                ],
            )

        graph_table_catalog = self._schema_graph.render_table_catalog()
        graph_relationship_catalog = (
            self._schema_graph.render_planning_relationship_catalog()
        )

        policy_for_prompt = (
            cleaned_policy
            if cleaned_policy
            else (
                "No policy evidence was supplied. Plan only evidence supported "
                "by the schema and the stated missing-evidence objective."
            )
        )

        existing_for_prompt = (
            cleaned_existing
            if cleaned_existing
            else "(no successful prior task metadata)"
        )

        base_prompt = _SUPPLEMENTAL_INVESTIGATION_PROMPT.format(
            supplemental_request=cleaned_request,
            original_query=cleaned_original or "(not supplied)",
            existing_evidence_summary=existing_for_prompt,
            ledger_context=_clean_text(ledger_context) or "(no ledger entries yet)",
            schema_context=cleaned_schema,
            graph_table_catalog=graph_table_catalog,
            graph_relationship_catalog=graph_relationship_catalog,
            policy_context=policy_for_prompt,
        )

        print(
            "[InvestigationPlanner]    Next-evidence mode: the current LLM "
            "missing-evidence objective is authoritative."
        )

        feedback: list[str] = []
        last_plan = InvestigationPlan(
            purpose=cleaned_request,
            is_investigation=True,
            tasks=[],
        )

        for attempt in range(1, MAX_SUPPLEMENTAL_PLAN_ATTEMPTS + 1):
            attempt_prompt = base_prompt

            if feedback:
                attempt_prompt += (
                    "\n\nPREVIOUS SUPPLEMENTAL PLAN FEEDBACK\n"
                    "-----------------------------------\n"
                    + "\n".join(f"- {item}" for item in feedback)
                    + "\n\nGenerate a materially revised supplemental plan. "
                    "Do not repeat the same evidence-domain mistake."
                )

            print(
                "[InvestigationPlanner]    Supplemental reasoning cycle "
                f"{attempt}/{MAX_SUPPLEMENTAL_PLAN_ATTEMPTS}"
            )

            generated = self._generate_plan(prompt=attempt_prompt)

            validated = self._validate_and_normalize_plan(
                generated,
                cleaned_query=cleaned_request,
                cleaned_schema=cleaned_schema,
            )

            last_plan = validated

            if not validated.tasks:
                reasons = validated.missing_business_rules or [
                    "No usable supplemental tasks survived validation."
                ]
                feedback = [
                    "Deterministic plan validation rejected the proposal: "
                    + "; ".join(reasons)
                ]
                print(
                    "[InvestigationPlanner]    Supplemental cycle rejected by "
                    "deterministic validation."
                )
                continue

            review = self._review_supplemental_plan(
                plan=validated,
                supplemental_request=cleaned_request,
                original_query=cleaned_original,
                existing_evidence_summary=existing_for_prompt,
                graph_table_catalog=graph_table_catalog,
                ledger_context=_clean_text(ledger_context),
            )

            print(
                "[InvestigationPlanner]    Supplemental self-review: "
                f"{'COMPLETE' if review.sufficient else 'PARTIAL'} — "
                f"{review.reason}"
            )

            if not review.sufficient:
                advisory = "Supplemental coverage gap: " + review.reason

                if advisory not in validated.missing_business_rules:
                    validated.missing_business_rules.append(advisory)

                if review.missing_requirements:
                    missing_note = "Remaining supplemental evidence: " + "; ".join(
                        review.missing_requirements
                    )
                    if missing_note not in validated.missing_business_rules:
                        validated.missing_business_rules.append(missing_note)

                # A plan the LLM itself says does not retrieve the requested
                # missing fact should not be executed immediately. Feed the
                # review back into the next bounded planning cycle.
                feedback = [
                    "The prior supplemental plan was judged insufficient.",
                    "Review reason: " + review.reason,
                ]

                if review.missing_requirements:
                    feedback.append(
                        "Still required: " + "; ".join(review.missing_requirements)
                    )

                if review.repeated_evidence_domains:
                    feedback.append(
                        "Avoid repeating already-covered evidence domains: "
                        + "; ".join(review.repeated_evidence_domains)
                    )

                if attempt < MAX_SUPPLEMENTAL_PLAN_ATTEMPTS:
                    print(
                        "[InvestigationPlanner]    Supplemental plan was PARTIAL; "
                        "using another bounded LLM planning cycle instead of "
                        "executing a knowingly inadequate task."
                    )
                    continue

                # At the final bounded attempt, preserve the limitation rather
                # than executing a plan that still fails its own evidence review.
                print(
                    "[InvestigationPlanner]    Final supplemental plan still "
                    "fails the missing-evidence objective; not returning its tasks."
                )
                validated.tasks = []
                return validated

            print(
                "[InvestigationPlanner]    Returning "
                f"{len(validated.tasks)} self-reviewed supplemental task(s)."
            )
            return validated

        limitations = list(last_plan.missing_business_rules)
        limitations.append(
            "The supplemental planner exhausted "
            f"{MAX_SUPPLEMENTAL_PLAN_ATTEMPTS} bounded reasoning cycles "
            "without producing a self-reviewed plan that closes the evidence gap."
        )

        last_plan.tasks = []
        last_plan.missing_business_rules = list(dict.fromkeys(limitations))

        print(
            "[InvestigationPlanner]    Supplemental planning stopped after "
            "bounded reasoning cycles without an acceptable evidence plan."
        )

        return last_plan


def _self_test_validated_plan_merge() -> None:
    """Verify validated plans merge beyond eight tasks up to the 16-task cap."""
    initial = InvestigationPlan(
        purpose="test",
        is_investigation=True,
        tasks=[
            InvestigationTask(
                task_id="T000",
                title="Initial",
                question="Initial operational evidence question?",
                scope=InvestigationScope.GENERAL,
                rationale="test",
            )
        ],
    )

    repaired_tasks = [
        InvestigationTask(
            task_id=f"T{i:03d}",
            title=f"Task {i}",
            question=f"Operational evidence question {i}?",
            scope=InvestigationScope.GENERAL,
            rationale="test",
        )
        for i in range(1, 17)
    ]

    repaired = InvestigationPlan(
        purpose="test",
        is_investigation=True,
        tasks=repaired_tasks,
    )

    merged = _merge_validated_investigation_plans(
        initial=initial,
        repaired=repaired,
    )

    assert len(merged.tasks) == MAX_INVESTIGATION_TASKS
    assert MAX_INVESTIGATION_TASKS == 16

    print("InvestigationPlanner 16-task merge self-test: PASS")
