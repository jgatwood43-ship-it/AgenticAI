# employees

## Purpose

Master employee identity, employment, department, job title, and contact data.

## Column dictionary

| Column | Data type | Nullable | Key | Extra | Description |
|---|---|---:|---|---|---|
| `employee_id` | `int` | NO | Primary key | auto_increment | Employee identifier. |
| `first_name` | `varchar` | NO | — | — | First name. |
| `middle_name` | `varchar` | YES | — | — | Optional middle name. |
| `last_name` | `varchar` | NO | — | — | Last name. |
| `date_of_birth` | `date` | NO | — | — | Date of birth. |
| `ssn_last_four` | `char` | NO | — | — | Last four SSN digits in anonymous test data. |
| `hire_date` | `date` | NO | — | — | Hire date. |
| `termination_date` | `date` | YES | — | — | Optional termination date. |
| `department_id` | `int` | YES | Non-unique index/key | — | Department identifier. |
| `job_title_id` | `int` | YES | Non-unique index/key | — | Job-title identifier. |
| `email` | `varchar` | YES | — | — | Email address. |
| `phone` | `varchar` | YES | — | — | Phone number. |
| `employment_type` | `varchar` | YES | — | — | Employment classification. |

## Relationships to every other table

Every other table is listed below so the file can stand alone for RAG/schema reasoning.

| Other table | Relationship type | Join possibility | Why it matters | Confidence |
|---|---|---|---|---|
| `badge_access_log` | Direct PK/FK | `badge_access_log.primary_employee_id = employees.employee_id OR badge_access_log.secondary_employee_id = employees.employee_id` | Primary and optional dual-custody participant. | High |
| `badge_access_rules` | Direct PK/FK | `badge_access_rules.employee_id = employees.employee_id` | Rules assigned to employee. | High |
| `departments` | Direct PK/FK | `employees.department_id = departments.department_id` | Employee department assignment. | High |
| `employee_status_history` | Direct PK/FK | `employee_status_history.employee_id = employees.employee_id` | Status history for employee. | High |
| `job_titles` | Direct PK/FK | `employees.job_title_id = job_titles.job_title_id` | Employee job-title assignment. | High |
| `key_access_log` | Direct PK/FK | `key_access_log.primary_employee_id = employees.employee_id OR key_access_log.secondary_employee_id = employees.employee_id` | Primary and optional dual-custody participant. | High |
| `key_inventory` | Indirect | `employee -> key_access_log participant -> key_inventory.key_id OR employee -> access rules -> room -> key_inventory` | Keys used by or relevant to employee. | High |
| `rooms` | Indirect | `employee -> badge_access_rules/badge_access_log/key_access_log -> rooms` | Rooms authorized for or accessed by employee. | High |
| `time_clock` | Direct PK/FK | `time_clock.employee_id = employees.employee_id` | Raw clock activity. | High |
| `time_clock_summary` | Direct PK/FK | `time_clock_summary.employee_id = employees.employee_id` | Daily time summary. | High |
| `video_cameras` | Indirect | `employee -> badge/rules room -> video_cameras.room_id` | Cameras covering employee-associated rooms. | High |
| `video_footage` | Indirect | `employee -> badge_access_log -> video_footage.badge_access_id` | Footage tied to employee badge events. | High |

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
