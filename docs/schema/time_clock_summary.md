# time_clock_summary

## Purpose

Daily employee time summaries derived from clock activity.

## Column dictionary

| Column | Data type | Nullable | Key | Extra | Description |
|---|---|---:|---|---|---|
| `summary_id` | `int` | NO | Primary key | auto_increment | Unique daily-summary ID. |
| `employee_id` | `int` | NO | Non-unique index/key | — | Employee identifier. |
| `work_date` | `date` | NO | — | — | Date summarized. |
| `clock_in_time` | `datetime` | YES | — | — | Clock-in timestamp. |
| `clock_out_time` | `datetime` | YES | — | — | Clock-out timestamp. |
| `hours_worked` | `decimal` | YES | — | — | Elapsed time between clock-in and clock-out, no meal/break deduction. |
| `status` | `varchar` | YES | — | — | Status value for the row. |

## Relationships to every other table

Every other table is listed below so the file can stand alone for RAG/schema reasoning.

| Other table | Relationship type | Join possibility | Why it matters | Confidence |
|---|---|---|---|---|
| `badge_access_log` | Temporal employee/day | `participant employee_id = time_clock_summary.employee_id AND DATE(access_time)=work_date` | Compare access with daily work interval. | High |
| `badge_access_rules` | Indirect employee | `badge_access_rules.employee_id = time_clock_summary.employee_id` | Compare authorization holders with daily work summaries. | High |
| `departments` | Indirect via employees | `departments.department_id = employees.department_id AND time_clock_summary.employee_id = employees.employee_id` | Daily summaries by department. | High |
| `employee_status_history` | Temporal employee | `employee_status_history.employee_id = time_clock_summary.employee_id AND effective_date <= work_date` | Daily work vs employment status. | High |
| `employees` | Direct PK/FK | `time_clock_summary.employee_id = employees.employee_id` | Daily time summary. | High |
| `job_titles` | Indirect | `job_titles -> employees -> time_clock_summary` | Daily work by role. | High |
| `key_access_log` | Temporal employee/day | `key participant = time_clock_summary.employee_id AND DATE(checked_out_at)=work_date` | Compare key use with daily work interval. | High |
| `key_inventory` | Indirect through key log | `key_inventory -> key_access_log participant/date -> time_clock_summary employee/work_date` | Daily work of key custodians. | High |
| `rooms` | Indirect | `room -> badge/key/rules employee -> time_clock_summary.employee_id` | Work summaries of people associated with room. | High |
| `time_clock` | Logical derivation | `time_clock.employee_id = time_clock_summary.employee_id AND DATE(punch_time)=work_date` | Raw punches to daily summary. | High |
| `video_cameras` | Indirect | `summary employee -> badge/key room -> video_cameras.room_id` | Cameras for employee activity during workday. | High |
| `video_footage` | Indirect temporal | `summary employee/date -> badge event -> video_footage` | Footage associated with workday access. | High |

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
