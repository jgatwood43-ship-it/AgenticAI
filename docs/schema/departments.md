# departments

## Purpose

Organizational departments and their primary location.

## Column dictionary

| Column | Data type | Nullable | Key | Extra | Description |
|---|---|---:|---|---|---|
| `department_id` | `int` | NO | Primary key | auto_increment | Department identifier. |
| `department_name` | `varchar` | NO | Unique index/key | — | Unique department name. |
| `location` | `varchar` | YES | — | — | Department location/building. |

## Relationships to every other table

Every other table is listed below so the file can stand alone for RAG/schema reasoning.

| Other table | Relationship type | Join possibility | Why it matters | Confidence |
|---|---|---|---|---|
| `badge_access_log` | No natural direct join | `No direct join recommended` | No reliable PK/FK, shared identifier, temporal key, or stable domain relationship is evident from this schema alone. Use an intermediate table only when business context requires it. | High |
| `badge_access_rules` | Indirect via employees | `badge_access_rules.employee_id = employees.employee_id AND employees.department_id = departments.department_id` | Analyze permissions by department. | High |
| `employee_status_history` | Indirect via employees | `departments.department_id = employees.department_id AND employee_status_history.employee_id = employees.employee_id` | Status history by department. | High |
| `employees` | Direct PK/FK | `employees.department_id = departments.department_id` | Employee department assignment. | High |
| `job_titles` | Domain/semantic | `departments.department_name ≈ job_titles.job_category` | Partial organizational-domain overlap only; do not treat as FK. | Medium |
| `key_access_log` | Indirect via employees | `departments.department_id = employees.department_id AND key participant = employees.employee_id` | Key activity by department. | High |
| `key_inventory` | Domain via building | `departments.location = rooms.building AND key_inventory.room_id = rooms.room_id` | Keys in department building; not authorization. | Medium |
| `rooms` | Domain location | `departments.location = rooms.building` | Shared building/location domain; not FK. | High |
| `time_clock` | Employee + domain | `departments.department_id = employees.department_id AND time_clock.employee_id = employees.employee_id; optionally departments.location = time_clock.location` | Compare employee punch location with home department location. | High |
| `time_clock_summary` | Indirect via employees | `departments.department_id = employees.department_id AND time_clock_summary.employee_id = employees.employee_id` | Daily summaries by department. | High |
| `video_cameras` | Domain via room | `departments.location = rooms.building AND video_cameras.room_id = rooms.room_id` | Cameras in department building. | Medium |
| `video_footage` | Indirect domain | `departments.location = rooms.building -> video_cameras -> video_footage` | Footage in department building. | Medium |

## Domain-related relationships

| Related table | Domain join/comparison | Meaning |
|---|---|---|
| `badge_access_log` | `No direct join recommended` | No reliable PK/FK, shared identifier, temporal key, or stable domain relationship is evident from this schema alone. Use an intermediate table only when business context requires it. |
| `job_titles` | `departments.department_name ≈ job_titles.job_category` | Partial organizational-domain overlap only; do not treat as FK. |
| `key_inventory` | `departments.location = rooms.building AND key_inventory.room_id = rooms.room_id` | Keys in department building; not authorization. |
| `rooms` | `departments.location = rooms.building` | Shared building/location domain; not FK. |
| `video_cameras` | `departments.location = rooms.building AND video_cameras.room_id = rooms.room_id` | Cameras in department building. |

## Join cautions

- `MUL` indicates a non-unique MySQL index/key; it does not prove that a FOREIGN KEY constraint is declared.
- Use direct PK/FK joins first; use domain and temporal joins only with the documented business context.
- Preserve primary vs secondary employee roles when analyzing dual custody.
- Do not join unrelated surrogate IDs simply because numeric values happen to match.
- Treat text-domain comparisons (for example building/location or access-level values) as semantic relationships, not referential integrity.

## Test-data note

The dataset is anonymous test data and may intentionally contain suspicious, incomplete, or inconsistent records for anomaly-detection and security-analysis testing.
