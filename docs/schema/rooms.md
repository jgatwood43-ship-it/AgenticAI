# rooms

## Purpose

Secured-room reference data including building, room type, dual custody, and access level.

## Column dictionary

| Column | Data type | Nullable | Key | Extra | Description |
|---|---|---:|---|---|---|
| `room_id` | `int` | NO | Primary key | auto_increment | Room identifier. |
| `room_name` | `varchar` | NO | Unique index/key | — | Unique room name. |
| `building` | `varchar` | NO | — | — | Building containing the room. |
| `room_type` | `varchar` | NO | — | — | Room classification. |
| `dual_custody_required` | `tinyint` | NO | — | — | 1/0 flag indicating whether two people are required. |
| `access_level` | `varchar` | NO | — | — | Required room access level. |

## Relationships to every other table

Every other table is listed below so the file can stand alone for RAG/schema reasoning.

| Other table | Relationship type | Join possibility | Why it matters | Confidence |
|---|---|---|---|---|
| `badge_access_log` | Direct PK/FK | `badge_access_log.room_id = rooms.room_id` | Badge event room. | High |
| `badge_access_rules` | Direct PK/FK | `badge_access_rules.room_id = rooms.room_id` | Rules assigned to room. | High |
| `departments` | Domain location | `departments.location = rooms.building` | Shared building/location domain; not FK. | High |
| `employee_status_history` | Indirect through access/rules | `status employee -> badge/key/rules -> rooms` | Rooms associated with employees by status period. | High |
| `employees` | Indirect | `employee -> badge_access_rules/badge_access_log/key_access_log -> rooms` | Rooms authorized for or accessed by employee. | High |
| `job_titles` | Indirect | `job_titles -> employees -> badge rules/events -> rooms` | Rooms associated with role. | High |
| `key_access_log` | Indirect via key | `key_access_log.key_id = key_inventory.key_id AND key_inventory.room_id = rooms.room_id` | Resolve key event to room. | High |
| `key_inventory` | Direct PK/FK (1:1 by UNIQUE) | `key_inventory.room_id = rooms.room_id` | Key to room; UNIQUE room_id means at most one key row per room. | High |
| `time_clock` | Domain location | `rooms.building = time_clock.location` | Shared building/location domain; not FK. | High |
| `time_clock_summary` | Indirect | `room -> badge/key/rules employee -> time_clock_summary.employee_id` | Work summaries of people associated with room. | High |
| `video_cameras` | Direct PK/FK | `video_cameras.room_id = rooms.room_id` | Camera coverage by room. | High |
| `video_footage` | Indirect via camera | `rooms.room_id = video_cameras.room_id AND video_footage.camera_id = video_cameras.camera_id` | All footage for a room. | High |

## Domain-related relationships

| Related table | Domain join/comparison | Meaning |
|---|---|---|
| `departments` | `departments.location = rooms.building` | Shared building/location domain; not FK. |
| `time_clock` | `rooms.building = time_clock.location` | Shared building/location domain; not FK. |

## Join cautions

- `MUL` indicates a non-unique MySQL index/key; it does not prove that a FOREIGN KEY constraint is declared.
- Use direct PK/FK joins first; use domain and temporal joins only with the documented business context.
- Preserve primary vs secondary employee roles when analyzing dual custody.
- Do not join unrelated surrogate IDs simply because numeric values happen to match.
- Treat text-domain comparisons (for example building/location or access-level values) as semantic relationships, not referential integrity.

## Test-data note

The dataset is anonymous test data and may intentionally contain suspicious, incomplete, or inconsistent records for anomaly-detection and security-analysis testing.
