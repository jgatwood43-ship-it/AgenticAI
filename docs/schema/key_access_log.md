# key_access_log

## Purpose

Physical-key custody/use events, including dual-custody participants and checkout/return times.

## Column dictionary

| Column | Data type | Nullable | Key | Extra | Description |
|---|---|---:|---|---|---|
| `access_id` | `int` | NO | Primary key | auto_increment | Unique key-access log ID. |
| `key_id` | `int` | NO | Non-unique index/key | — | Physical key identifier. |
| `primary_employee_id` | `int` | NO | Non-unique index/key | — | Primary employee participant. |
| `secondary_employee_id` | `int` | YES | Non-unique index/key | — | Optional second employee for dual custody. |
| `checked_out_at` | `datetime` | NO | — | DEFAULT_GENERATED | Key checkout timestamp. |
| `returned_at` | `datetime` | YES | — | — | Return timestamp, nullable. |
| `purpose` | `varchar` | YES | — | — | Business purpose. |
| `status` | `varchar` | NO | — | — | Status value for the row. |

## Relationships to every other table

Every other table is listed below so the file can stand alone for RAG/schema reasoning.

| Other table | Relationship type | Join possibility | Why it matters | Confidence |
|---|---|---|---|---|
| `badge_access_log` | Indirect temporal/room | `badge_access_log.room_id = key_inventory.room_id; key_access_log.key_id = key_inventory.key_id; match participant + nearby/overlapping time` | Correlate badge and key activity. | High |
| `badge_access_rules` | Logical authorization | `rule room = key_inventory.room_id via key_access_log.key_id; participant employee = rule employee; checkout date within validity` | Validate key custodian authorization. | High |
| `departments` | Indirect via employees | `departments.department_id = employees.department_id AND key participant = employees.employee_id` | Key activity by department. | High |
| `employee_status_history` | Temporal employee | `key participant = employee_status_history.employee_id AND effective_date <= DATE(checked_out_at)` | Employment status at key checkout. | High |
| `employees` | Direct PK/FK | `key_access_log.primary_employee_id = employees.employee_id OR key_access_log.secondary_employee_id = employees.employee_id` | Primary and optional dual-custody participant. | High |
| `job_titles` | Indirect via employees | `job_titles.job_title_id = employees.job_title_id AND key participant = employees.employee_id` | Key activity by role. | High |
| `key_inventory` | Direct PK/FK | `key_access_log.key_id = key_inventory.key_id` | Key event to physical key. | High |
| `rooms` | Indirect via key | `key_access_log.key_id = key_inventory.key_id AND key_inventory.room_id = rooms.room_id` | Resolve key event to room. | High |
| `time_clock` | Temporal employee | `key participant = time_clock.employee_id; compare checkout/return with punch interval` | Check key use while on duty. | High |
| `time_clock_summary` | Temporal employee/day | `key participant = time_clock_summary.employee_id AND DATE(checked_out_at)=work_date` | Compare key use with daily work interval. | High |
| `video_cameras` | Indirect room | `key_access_log.key_id -> key_inventory.room_id = video_cameras.room_id` | Find cameras covering key-associated room. | High |
| `video_footage` | Indirect room + time | `key -> room -> camera -> footage; compare footage interval with checkout/return interval` | Corroborate key activity with video. | High |

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
