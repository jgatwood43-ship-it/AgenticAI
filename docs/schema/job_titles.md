# job_titles

## Purpose

Reference list of job titles and broader job categories.

## Column dictionary

| Column | Data type | Nullable | Key | Extra | Description |
|---|---|---:|---|---|---|
| `job_title_id` | `int` | NO | Primary key | auto_increment | Job-title identifier. |
| `job_title_name` | `varchar` | NO | Unique index/key | — | Unique job-title name. |
| `job_category` | `varchar` | YES | — | — | Broader job family/category. |

## Relationships to every other table

Every other table is listed below so the file can stand alone for RAG/schema reasoning.

| Other table | Relationship type | Join possibility | Why it matters | Confidence |
|---|---|---|---|---|
| `badge_access_log` | No natural direct join | `No direct join recommended` | No reliable PK/FK, shared identifier, temporal key, or stable domain relationship is evident from this schema alone. Use an intermediate table only when business context requires it. | High |
| `badge_access_rules` | Indirect via employees | `badge_access_rules.employee_id = employees.employee_id AND employees.job_title_id = job_titles.job_title_id` | Analyze permissions by job role. | High |
| `departments` | Domain/semantic | `departments.department_name ≈ job_titles.job_category` | Partial organizational-domain overlap only; do not treat as FK. | Medium |
| `employee_status_history` | Indirect via employees | `employee_status_history.employee_id = employees.employee_id AND employees.job_title_id = job_titles.job_title_id` | Status history by role. | High |
| `employees` | Direct PK/FK | `employees.job_title_id = job_titles.job_title_id` | Employee job-title assignment. | High |
| `key_access_log` | Indirect via employees | `job_titles.job_title_id = employees.job_title_id AND key participant = employees.employee_id` | Key activity by role. | High |
| `key_inventory` | Indirect | `job_titles -> employees -> key_access_log -> key_inventory` | Keys used by role. | High |
| `rooms` | Indirect | `job_titles -> employees -> badge rules/events -> rooms` | Rooms associated with role. | High |
| `time_clock` | Indirect | `job_titles -> employees -> time_clock` | Clock activity by role. | High |
| `time_clock_summary` | Indirect | `job_titles -> employees -> time_clock_summary` | Daily work by role. | High |
| `video_cameras` | Indirect | `job_titles -> employees -> access room -> video_cameras` | Camera coverage of role-associated rooms. | High |
| `video_footage` | Indirect | `job_titles -> employees -> badge_access_log -> video_footage` | Footage tied to role activity. | High |

## Domain-related relationships

| Related table | Domain join/comparison | Meaning |
|---|---|---|
| `badge_access_log` | `No direct join recommended` | No reliable PK/FK, shared identifier, temporal key, or stable domain relationship is evident from this schema alone. Use an intermediate table only when business context requires it. |
| `departments` | `departments.department_name ≈ job_titles.job_category` | Partial organizational-domain overlap only; do not treat as FK. |

## Join cautions

- `MUL` indicates a non-unique MySQL index/key; it does not prove that a FOREIGN KEY constraint is declared.
- Use direct PK/FK joins first; use domain and temporal joins only with the documented business context.
- Preserve primary vs secondary employee roles when analyzing dual custody.
- Do not join unrelated surrogate IDs simply because numeric values happen to match.
- Treat text-domain comparisons (for example building/location or access-level values) as semantic relationships, not referential integrity.

## Test-data note

The dataset is anonymous test data and may intentionally contain suspicious, incomplete, or inconsistent records for anomaly-detection and security-analysis testing.
