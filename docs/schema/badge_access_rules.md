# badge_access_rules

## Purpose

Employee-to-room authorization rules with access level and effective dates.

## Column dictionary

| Column | Data type | Nullable | Key | Extra | Description |
|---|---|---:|---|---|---|
| `rule_id` | `int` | NO | Primary key | auto_increment | Unique access-rule ID. |
| `employee_id` | `int` | NO | Non-unique index/key | — | Employee identifier. |
| `room_id` | `int` | NO | Non-unique index/key | — | Room identifier. |
| `access_level_granted` | `varchar` | NO | — | — | Access level granted by the rule. |
| `valid_from` | `date` | NO | — | — | First valid date. |
| `valid_until` | `date` | YES | — | — | Inclusive final valid date; NULL means open-ended. |

## Relationships to every other table

Every other table is listed below so the file can stand alone for RAG/schema reasoning.

| Other table | Relationship type | Join possibility | Why it matters | Confidence |
|---|---|---|---|---|
| `badge_access_log` | Logical authorization | `badge_access_log.room_id = badge_access_rules.room_id AND participant employee_id = badge_access_rules.employee_id AND DATE(access_time) BETWEEN valid_from AND COALESCE(valid_until,'9999-12-31')` | Point-in-time authorization check. | High |
| `departments` | Indirect via employees | `badge_access_rules.employee_id = employees.employee_id AND employees.department_id = departments.department_id` | Analyze permissions by department. | High |
| `employee_status_history` | Temporal employee | `badge_access_rules.employee_id = employee_status_history.employee_id; compare effective_date to rule validity` | Detect permissions persisting through status changes. | High |
| `employees` | Direct PK/FK | `badge_access_rules.employee_id = employees.employee_id` | Rules assigned to employee. | High |
| `job_titles` | Indirect via employees | `badge_access_rules.employee_id = employees.employee_id AND employees.job_title_id = job_titles.job_title_id` | Analyze permissions by job role. | High |
| `key_access_log` | Logical authorization | `rule room = key_inventory.room_id via key_access_log.key_id; participant employee = rule employee; checkout date within validity` | Validate key custodian authorization. | High |
| `key_inventory` | Room domain | `badge_access_rules.room_id = key_inventory.room_id` | Find key for authorized room. | High |
| `rooms` | Direct PK/FK | `badge_access_rules.room_id = rooms.room_id` | Rules assigned to room. | High |
| `time_clock` | Indirect employee | `badge_access_rules.employee_id = time_clock.employee_id` | Compare authorization holders with work activity. | High |
| `time_clock_summary` | Indirect employee | `badge_access_rules.employee_id = time_clock_summary.employee_id` | Compare authorization holders with daily work summaries. | High |
| `video_cameras` | Room domain | `badge_access_rules.room_id = video_cameras.room_id` | Find cameras for authorized rooms. | High |
| `video_footage` | Indirect room/event | `rule room -> video_cameras.room_id -> video_footage.camera_id, or through badge_access_log` | Find video tied to governed rooms/events. | High |

## Domain-related relationships

No standalone domain relationship is evident beyond identifier, temporal, or indirect joins documented above.

## Join cautions

- `MUL` indicates a non-unique MySQL index/key; it does not prove that a FOREIGN KEY constraint is declared.
- Use direct PK/FK joins first; use domain and temporal joins only with the documented business context.
- Preserve primary vs secondary employee roles when analyzing dual custody.
- Do not join unrelated surrogate IDs simply because numeric values happen to match.
- Treat text-domain comparisons (for example building/location or access-level values) as semantic relationships, not referential integrity.

## Test-data note

The dataset is anonymous test data and may intentionally contain suspicious, incomplete, or inconsistent records for anomaly-detection and security-analysis testing.
