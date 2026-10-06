# employee_status_history

## Purpose

Historical employee employment-status changes.

## Column dictionary

| Column | Data type | Nullable | Key | Extra | Description |
|---|---|---:|---|---|---|
| `status_id` | `int` | NO | Primary key | auto_increment | Unique status-history ID. |
| `employee_id` | `int` | NO | Non-unique index/key | — | Employee identifier. |
| `status` | `varchar` | NO | — | — | Status value for the row. |
| `effective_date` | `date` | NO | — | — | Date the status became effective. |
| `changed_by` | `varchar` | YES | — | — | Free-text actor; not a foreign key. |
| `notes` | `text` | YES | — | — | Optional notes. |

## Relationships to every other table

Every other table is listed below so the file can stand alone for RAG/schema reasoning.

| Other table | Relationship type | Join possibility | Why it matters | Confidence |
|---|---|---|---|---|
| `badge_access_log` | Temporal | `participant employee_id = employee_status_history.employee_id AND effective_date <= DATE(access_time)` | Find latest effective employment status at access time. | High |
| `badge_access_rules` | Temporal employee | `badge_access_rules.employee_id = employee_status_history.employee_id; compare effective_date to rule validity` | Detect permissions persisting through status changes. | High |
| `departments` | Indirect via employees | `departments.department_id = employees.department_id AND employee_status_history.employee_id = employees.employee_id` | Status history by department. | High |
| `employees` | Direct PK/FK | `employee_status_history.employee_id = employees.employee_id` | Status history for employee. | High |
| `job_titles` | Indirect via employees | `employee_status_history.employee_id = employees.employee_id AND employees.job_title_id = job_titles.job_title_id` | Status history by role. | High |
| `key_access_log` | Temporal employee | `key participant = employee_status_history.employee_id AND effective_date <= DATE(checked_out_at)` | Employment status at key checkout. | High |
| `key_inventory` | Indirect through key log | `status employee -> key_access_log participant -> key_inventory.key_id` | Keys used during a status period. | High |
| `rooms` | Indirect through access/rules | `status employee -> badge/key/rules -> rooms` | Rooms associated with employees by status period. | High |
| `time_clock` | Temporal employee | `employee_status_history.employee_id = time_clock.employee_id AND effective_date <= DATE(punch_time)` | Clock activity vs employment status. | High |
| `time_clock_summary` | Temporal employee | `employee_status_history.employee_id = time_clock_summary.employee_id AND effective_date <= work_date` | Daily work vs employment status. | High |
| `video_cameras` | Indirect | `status employee -> badge event room -> video_cameras.room_id` | Cameras covering rooms accessed during status period. | High |
| `video_footage` | Indirect temporal | `status employee -> badge_access_log -> video_footage.badge_access_id` | Footage for employee access during status period. | High |

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
