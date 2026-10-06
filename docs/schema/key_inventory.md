# key_inventory

## Purpose

Inventory of physical keys and the room each key serves.

## Column dictionary

| Column | Data type | Nullable | Key | Extra | Description |
|---|---|---:|---|---|---|
| `key_id` | `int` | NO | Primary key | auto_increment | Physical key identifier. |
| `room_id` | `int` | NO | Unique index/key | — | Room identifier. |
| `key_code` | `varchar` | NO | Unique index/key | — | Unique key code. |
| `key_status` | `varchar` | NO | — | — | Inventory status. |
| `notes` | `text` | YES | — | — | Optional notes. |

## Relationships to every other table

Every other table is listed below so the file can stand alone for RAG/schema reasoning.

| Other table | Relationship type | Join possibility | Why it matters | Confidence |
|---|---|---|---|---|
| `badge_access_log` | Indirect room | `badge_access_log.room_id = key_inventory.room_id` | Find key associated with accessed room. | High |
| `badge_access_rules` | Room domain | `badge_access_rules.room_id = key_inventory.room_id` | Find key for authorized room. | High |
| `departments` | Domain via building | `departments.location = rooms.building AND key_inventory.room_id = rooms.room_id` | Keys in department building; not authorization. | Medium |
| `employee_status_history` | Indirect through key log | `status employee -> key_access_log participant -> key_inventory.key_id` | Keys used during a status period. | High |
| `employees` | Indirect | `employee -> key_access_log participant -> key_inventory.key_id OR employee -> access rules -> room -> key_inventory` | Keys used by or relevant to employee. | High |
| `job_titles` | Indirect | `job_titles -> employees -> key_access_log -> key_inventory` | Keys used by role. | High |
| `key_access_log` | Direct PK/FK | `key_access_log.key_id = key_inventory.key_id` | Key event to physical key. | High |
| `rooms` | Direct PK/FK (1:1 by UNIQUE) | `key_inventory.room_id = rooms.room_id` | Key to room; UNIQUE room_id means at most one key row per room. | High |
| `time_clock` | Domain location | `key_inventory.room_id -> rooms.building = time_clock.location` | Clock activity in key room building. | Medium |
| `time_clock_summary` | Indirect through key log | `key_inventory -> key_access_log participant/date -> time_clock_summary employee/work_date` | Daily work of key custodians. | High |
| `video_cameras` | Room domain | `key_inventory.room_id = video_cameras.room_id` | Cameras for key-associated room. | High |
| `video_footage` | Indirect room/camera | `key_inventory.room_id = video_cameras.room_id AND video_footage.camera_id = video_cameras.camera_id` | Footage for key-associated room. | High |

## Domain-related relationships

| Related table | Domain join/comparison | Meaning |
|---|---|---|
| `departments` | `departments.location = rooms.building AND key_inventory.room_id = rooms.room_id` | Keys in department building; not authorization. |
| `time_clock` | `key_inventory.room_id -> rooms.building = time_clock.location` | Clock activity in key room building. |

## Join cautions

- `MUL` indicates a non-unique MySQL index/key; it does not prove that a FOREIGN KEY constraint is declared.
- Use direct PK/FK joins first; use domain and temporal joins only with the documented business context.
- Preserve primary vs secondary employee roles when analyzing dual custody.
- Do not join unrelated surrogate IDs simply because numeric values happen to match.
- Treat text-domain comparisons (for example building/location or access-level values) as semantic relationships, not referential integrity.

## Test-data note

The dataset is anonymous test data and may intentionally contain suspicious, incomplete, or inconsistent records for anomaly-detection and security-analysis testing.
