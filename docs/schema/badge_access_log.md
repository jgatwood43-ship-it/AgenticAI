# badge_access_log

## Purpose

Physical badge-access events by room, employee participant(s), time, result, and direction.

## Column dictionary

| Column | Data type | Nullable | Key | Extra | Description |
|---|---|---:|---|---|---|
| `badge_access_id` | `int` | NO | Primary key | auto_increment | Optional linked badge-event ID. |
| `room_id` | `int` | NO | Non-unique index/key | — | Room identifier. |
| `primary_employee_id` | `int` | NO | Non-unique index/key | — | Primary employee participant. |
| `secondary_employee_id` | `int` | YES | Non-unique index/key | — | Optional second employee for dual custody. |
| `access_time` | `datetime` | NO | — | DEFAULT_GENERATED | Event timestamp. |
| `access_result` | `varchar` | NO | — | — | Event outcome/classification; e.g. Granted, Denied, Tailgate. |
| `direction` | `varchar` | NO | — | — | Movement direction, typically IN/OUT. |

## Relationships to every other table

Every other table is listed below so the file can stand alone for RAG/schema reasoning.

| Other table | Relationship type | Join possibility | Why it matters | Confidence |
|---|---|---|---|---|
| `badge_access_rules` | Logical authorization | `badge_access_log.room_id = badge_access_rules.room_id AND participant employee_id = badge_access_rules.employee_id AND DATE(access_time) BETWEEN valid_from AND COALESCE(valid_until,'9999-12-31')` | Point-in-time authorization check. | High |
| `departments` | No natural direct join | `No direct join recommended` | No reliable PK/FK, shared identifier, temporal key, or stable domain relationship is evident from this schema alone. Use an intermediate table only when business context requires it. | High |
| `employee_status_history` | Temporal | `participant employee_id = employee_status_history.employee_id AND effective_date <= DATE(access_time)` | Find latest effective employment status at access time. | High |
| `employees` | Direct PK/FK | `badge_access_log.primary_employee_id = employees.employee_id OR badge_access_log.secondary_employee_id = employees.employee_id` | Primary and optional dual-custody participant. | High |
| `job_titles` | No natural direct join | `No direct join recommended` | No reliable PK/FK, shared identifier, temporal key, or stable domain relationship is evident from this schema alone. Use an intermediate table only when business context requires it. | High |
| `key_access_log` | Indirect temporal/room | `badge_access_log.room_id = key_inventory.room_id; key_access_log.key_id = key_inventory.key_id; match participant + nearby/overlapping time` | Correlate badge and key activity. | High |
| `key_inventory` | Indirect room | `badge_access_log.room_id = key_inventory.room_id` | Find key associated with accessed room. | High |
| `rooms` | Direct PK/FK | `badge_access_log.room_id = rooms.room_id` | Badge event room. | High |
| `time_clock` | Temporal employee/location | `participant employee_id = time_clock.employee_id; compare access_time to punch_time and room building to punch location` | Check whether access occurred while on duty/on site. | High |
| `time_clock_summary` | Temporal employee/day | `participant employee_id = time_clock_summary.employee_id AND DATE(access_time)=work_date` | Compare access with daily work interval. | High |
| `video_cameras` | Indirect room | `badge_access_log.room_id = video_cameras.room_id` | Find cameras covering badge-event room. | High |
| `video_footage` | Direct PK/FK (optional) | `video_footage.badge_access_id = badge_access_log.badge_access_id` | Optional event-footage link. | High |

## Domain-related relationships

| Related table | Domain join/comparison | Meaning |
|---|---|---|
| `departments` | `No direct join recommended` | No reliable PK/FK, shared identifier, temporal key, or stable domain relationship is evident from this schema alone. Use an intermediate table only when business context requires it. |
| `job_titles` | `No direct join recommended` | No reliable PK/FK, shared identifier, temporal key, or stable domain relationship is evident from this schema alone. Use an intermediate table only when business context requires it. |

## Join cautions

- `MUL` indicates a non-unique MySQL index/key; it does not prove that a FOREIGN KEY constraint is declared.
- Use direct PK/FK joins first; use domain and temporal joins only with the documented business context.
- Preserve primary vs secondary employee roles when analyzing dual custody.
- Do not join unrelated surrogate IDs simply because numeric values happen to match.
- Treat text-domain comparisons (for example building/location or access-level values) as semantic relationships, not referential integrity.

## Test-data note

The dataset is anonymous test data and may intentionally contain suspicious, incomplete, or inconsistent records for anomaly-detection and security-analysis testing.
