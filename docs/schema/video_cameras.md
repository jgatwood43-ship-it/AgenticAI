# video_cameras

## Purpose

Surveillance camera inventory and room assignment.

## Column dictionary

| Column | Data type | Nullable | Key | Extra | Description |
|---|---|---:|---|---|---|
| `camera_id` | `int` | NO | Primary key | auto_increment | Camera identifier. |
| `room_id` | `int` | NO | Non-unique index/key | — | Room identifier. |
| `camera_name` | `varchar` | NO | — | — | Camera name. |
| `position` | `varchar` | NO | — | — | Camera placement. |
| `status` | `varchar` | NO | — | — | Status value for the row. |

## Relationships to every other table

Every other table is listed below so the file can stand alone for RAG/schema reasoning.

| Other table | Relationship type | Join possibility | Why it matters | Confidence |
|---|---|---|---|---|
| `badge_access_log` | Indirect room | `badge_access_log.room_id = video_cameras.room_id` | Find cameras covering badge-event room. | High |
| `badge_access_rules` | Room domain | `badge_access_rules.room_id = video_cameras.room_id` | Find cameras for authorized rooms. | High |
| `departments` | Domain via room | `departments.location = rooms.building AND video_cameras.room_id = rooms.room_id` | Cameras in department building. | Medium |
| `employee_status_history` | Indirect | `status employee -> badge event room -> video_cameras.room_id` | Cameras covering rooms accessed during status period. | High |
| `employees` | Indirect | `employee -> badge/rules room -> video_cameras.room_id` | Cameras covering employee-associated rooms. | High |
| `job_titles` | Indirect | `job_titles -> employees -> access room -> video_cameras` | Camera coverage of role-associated rooms. | High |
| `key_access_log` | Indirect room | `key_access_log.key_id -> key_inventory.room_id = video_cameras.room_id` | Find cameras covering key-associated room. | High |
| `key_inventory` | Room domain | `key_inventory.room_id = video_cameras.room_id` | Cameras for key-associated room. | High |
| `rooms` | Direct PK/FK | `video_cameras.room_id = rooms.room_id` | Camera coverage by room. | High |
| `time_clock` | Domain via building | `time_clock.location = rooms.building AND video_cameras.room_id = rooms.room_id` | Cameras at punch building. | Medium |
| `time_clock_summary` | Indirect | `summary employee -> badge/key room -> video_cameras.room_id` | Cameras for employee activity during workday. | High |
| `video_footage` | Direct PK/FK | `video_footage.camera_id = video_cameras.camera_id` | Footage to recording camera. | High |

## Domain-related relationships

| Related table | Domain join/comparison | Meaning |
|---|---|---|
| `departments` | `departments.location = rooms.building AND video_cameras.room_id = rooms.room_id` | Cameras in department building. |
| `time_clock` | `time_clock.location = rooms.building AND video_cameras.room_id = rooms.room_id` | Cameras at punch building. |

## Join cautions

- `MUL` indicates a non-unique MySQL index/key; it does not prove that a FOREIGN KEY constraint is declared.
- Use direct PK/FK joins first; use domain and temporal joins only with the documented business context.
- Preserve primary vs secondary employee roles when analyzing dual custody.
- Do not join unrelated surrogate IDs simply because numeric values happen to match.
- Treat text-domain comparisons (for example building/location or access-level values) as semantic relationships, not referential integrity.

## Test-data note

The dataset is anonymous test data and may intentionally contain suspicious, incomplete, or inconsistent records for anomaly-detection and security-analysis testing.
