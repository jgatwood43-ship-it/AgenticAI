# time_clock

## Purpose

Raw employee IN/OUT time-clock punches and punch location.

## Column dictionary

| Column | Data type | Nullable | Key | Extra | Description |
|---|---|---:|---|---|---|
| `clock_id` | `int` | NO | Primary key | auto_increment | Unique clock-punch ID. |
| `employee_id` | `int` | NO | Non-unique index/key | — | Employee identifier. |
| `punch_type` | `varchar` | NO | — | — | IN/OUT punch type. |
| `punch_time` | `datetime` | NO | — | DEFAULT_GENERATED | Punch timestamp. |
| `location` | `varchar` | YES | — | — | Department location/building. |
| `notes` | `text` | YES | — | — | Optional notes. |

## Relationships to every other table

Every other table is listed below so the file can stand alone for RAG/schema reasoning.

| Other table | Relationship type | Join possibility | Why it matters | Confidence |
|---|---|---|---|---|
| `badge_access_log` | Temporal employee/location | `participant employee_id = time_clock.employee_id; compare access_time to punch_time and room building to punch location` | Check whether access occurred while on duty/on site. | High |
| `badge_access_rules` | Indirect employee | `badge_access_rules.employee_id = time_clock.employee_id` | Compare authorization holders with work activity. | High |
| `departments` | Employee + domain | `departments.department_id = employees.department_id AND time_clock.employee_id = employees.employee_id; optionally departments.location = time_clock.location` | Compare employee punch location with home department location. | High |
| `employee_status_history` | Temporal employee | `employee_status_history.employee_id = time_clock.employee_id AND effective_date <= DATE(punch_time)` | Clock activity vs employment status. | High |
| `employees` | Direct PK/FK | `time_clock.employee_id = employees.employee_id` | Raw clock activity. | High |
| `job_titles` | Indirect | `job_titles -> employees -> time_clock` | Clock activity by role. | High |
| `key_access_log` | Temporal employee | `key participant = time_clock.employee_id; compare checkout/return with punch interval` | Check key use while on duty. | High |
| `key_inventory` | Domain location | `key_inventory.room_id -> rooms.building = time_clock.location` | Clock activity in key room building. | Medium |
| `rooms` | Domain location | `rooms.building = time_clock.location` | Shared building/location domain; not FK. | High |
| `time_clock_summary` | Logical derivation | `time_clock.employee_id = time_clock_summary.employee_id AND DATE(punch_time)=work_date` | Raw punches to daily summary. | High |
| `video_cameras` | Domain via building | `time_clock.location = rooms.building AND video_cameras.room_id = rooms.room_id` | Cameras at punch building. | Medium |
| `video_footage` | Domain + temporal | `time_clock.location = rooms.building -> cameras -> footage; compare punch_time to footage interval` | Footage near punch location/time. | Medium |

## Domain-related relationships

| Related table | Domain join/comparison | Meaning |
|---|---|---|
| `key_inventory` | `key_inventory.room_id -> rooms.building = time_clock.location` | Clock activity in key room building. |
| `rooms` | `rooms.building = time_clock.location` | Shared building/location domain; not FK. |
| `video_cameras` | `time_clock.location = rooms.building AND video_cameras.room_id = rooms.room_id` | Cameras at punch building. |
| `video_footage` | `time_clock.location = rooms.building -> cameras -> footage; compare punch_time to footage interval` | Footage near punch location/time. |

## Join cautions

- `MUL` indicates a non-unique MySQL index/key; it does not prove that a FOREIGN KEY constraint is declared.
- Use direct PK/FK joins first; use domain and temporal joins only with the documented business context.
- Preserve primary vs secondary employee roles when analyzing dual custody.
- Do not join unrelated surrogate IDs simply because numeric values happen to match.
- Treat text-domain comparisons (for example building/location or access-level values) as semantic relationships, not referential integrity.

## Test-data note

The dataset is anonymous test data and may intentionally contain suspicious, incomplete, or inconsistent records for anomaly-detection and security-analysis testing.
