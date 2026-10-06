# video_footage

## Purpose

Stored surveillance-footage segments, optionally linked to badge events.

## Column dictionary

| Column | Data type | Nullable | Key | Extra | Description |
|---|---|---:|---|---|---|
| `footage_id` | `int` | NO | Primary key | auto_increment | Unique footage ID. |
| `camera_id` | `int` | NO | Non-unique index/key | — | Camera identifier. |
| `badge_access_id` | `int` | YES | Non-unique index/key | — | Optional linked badge-event ID. |
| `start_time` | `datetime` | NO | — | — | Footage start time. |
| `end_time` | `datetime` | YES | — | — | Footage end time. |
| `storage_path` | `varchar` | NO | — | — | Stored footage path. |
| `footage_type` | `varchar` | NO | — | — | Footage classification. |

## Relationships to every other table

Every other table is listed below so the file can stand alone for RAG/schema reasoning.

| Other table | Relationship type | Join possibility | Why it matters | Confidence |
|---|---|---|---|---|
| `badge_access_log` | Direct PK/FK (optional) | `video_footage.badge_access_id = badge_access_log.badge_access_id` | Optional event-footage link. | High |
| `badge_access_rules` | Indirect room/event | `rule room -> video_cameras.room_id -> video_footage.camera_id, or through badge_access_log` | Find video tied to governed rooms/events. | High |
| `departments` | Indirect domain | `departments.location = rooms.building -> video_cameras -> video_footage` | Footage in department building. | Medium |
| `employee_status_history` | Indirect temporal | `status employee -> badge_access_log -> video_footage.badge_access_id` | Footage for employee access during status period. | High |
| `employees` | Indirect | `employee -> badge_access_log -> video_footage.badge_access_id` | Footage tied to employee badge events. | High |
| `job_titles` | Indirect | `job_titles -> employees -> badge_access_log -> video_footage` | Footage tied to role activity. | High |
| `key_access_log` | Indirect room + time | `key -> room -> camera -> footage; compare footage interval with checkout/return interval` | Corroborate key activity with video. | High |
| `key_inventory` | Indirect room/camera | `key_inventory.room_id = video_cameras.room_id AND video_footage.camera_id = video_cameras.camera_id` | Footage for key-associated room. | High |
| `rooms` | Indirect via camera | `rooms.room_id = video_cameras.room_id AND video_footage.camera_id = video_cameras.camera_id` | All footage for a room. | High |
| `time_clock` | Domain + temporal | `time_clock.location = rooms.building -> cameras -> footage; compare punch_time to footage interval` | Footage near punch location/time. | Medium |
| `time_clock_summary` | Indirect temporal | `summary employee/date -> badge event -> video_footage` | Footage associated with workday access. | High |
| `video_cameras` | Direct PK/FK | `video_footage.camera_id = video_cameras.camera_id` | Footage to recording camera. | High |

## Domain-related relationships

| Related table | Domain join/comparison | Meaning |
|---|---|---|
| `time_clock` | `time_clock.location = rooms.building -> cameras -> footage; compare punch_time to footage interval` | Footage near punch location/time. |

## Join cautions

- `MUL` indicates a non-unique MySQL index/key; it does not prove that a FOREIGN KEY constraint is declared.
- Use direct PK/FK joins first; use domain and temporal joins only with the documented business context.
- Preserve primary vs secondary employee roles when analyzing dual custody.
- Do not join unrelated surrogate IDs simply because numeric values happen to match.
- Treat text-domain comparisons (for example building/location or access-level values) as semantic relationships, not referential integrity.

## Test-data note

The dataset is anonymous test data and may intentionally contain suspicious, incomplete, or inconsistent records for anomaly-detection and security-analysis testing.
