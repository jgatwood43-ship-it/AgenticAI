# Security Database Relationship Guide

This guide documents all direct, indirect, temporal, and domain-related relationships among the 13 tables.

## Domain map

| Domain | Columns / tables | Relationship meaning |
|---|---|---|
| Employee identity | `employees.employee_id`; employee IDs in access, key, time, status, and rules tables | Main person-centric join domain. |
| Room identity | `rooms.room_id`; room IDs in badge rules/log, key inventory, and cameras | Main physical-space join domain. |
| Building/location | `rooms.building`, `departments.location`, `time_clock.location` | Shared semantic location domain; not an FK. |
| Access level | `rooms.access_level`, `badge_access_rules.access_level_granted` | Required-vs-granted access comparison. |
| Employment state | `employee_status_history.status` plus employee hire/termination dates | Point-in-time employment-state analysis. |
| Event time | badge times, key checkout/return, clock punches, summary intervals, footage intervals | Cross-system temporal correlation. |
| Dual custody | `rooms.dual_custody_required`, badge/key `secondary_employee_id` | Determines whether a second participant is expected. |
| Organizational role | `departments.department_name`, `job_titles.job_category` | Partial semantic overlap only; not a reliable FK. |
| Status-like text | badge result, key status, time-summary status, camera status, key inventory status | Different domains despite similar data type; do not join by text equality. |

## Exhaustive pairwise relationship matrix

There are 78 unique table-to-table pairs; every pair is listed once.

| Table A | Table B | Relationship type | Join possibility | Notes |
|---|---|---|---|---|
| `badge_access_log` | `badge_access_rules` | Logical authorization | `badge_access_log.room_id = badge_access_rules.room_id AND participant employee_id = badge_access_rules.employee_id AND DATE(access_time) BETWEEN valid_from AND COALESCE(valid_until,'9999-12-31')` | Point-in-time authorization check. |
| `badge_access_log` | `departments` | No natural direct join | `No direct join recommended` | No reliable PK/FK, shared identifier, temporal key, or stable domain relationship is evident from this schema alone. Use an intermediate table only when business context requires it. |
| `badge_access_log` | `employee_status_history` | Temporal | `participant employee_id = employee_status_history.employee_id AND effective_date <= DATE(access_time)` | Find latest effective employment status at access time. |
| `badge_access_log` | `employees` | Direct PK/FK | `badge_access_log.primary_employee_id = employees.employee_id OR badge_access_log.secondary_employee_id = employees.employee_id` | Primary and optional dual-custody participant. |
| `badge_access_log` | `job_titles` | No natural direct join | `No direct join recommended` | No reliable PK/FK, shared identifier, temporal key, or stable domain relationship is evident from this schema alone. Use an intermediate table only when business context requires it. |
| `badge_access_log` | `key_access_log` | Indirect temporal/room | `badge_access_log.room_id = key_inventory.room_id; key_access_log.key_id = key_inventory.key_id; match participant + nearby/overlapping time` | Correlate badge and key activity. |
| `badge_access_log` | `key_inventory` | Indirect room | `badge_access_log.room_id = key_inventory.room_id` | Find key associated with accessed room. |
| `badge_access_log` | `rooms` | Direct PK/FK | `badge_access_log.room_id = rooms.room_id` | Badge event room. |
| `badge_access_log` | `time_clock` | Temporal employee/location | `participant employee_id = time_clock.employee_id; compare access_time to punch_time and room building to punch location` | Check whether access occurred while on duty/on site. |
| `badge_access_log` | `time_clock_summary` | Temporal employee/day | `participant employee_id = time_clock_summary.employee_id AND DATE(access_time)=work_date` | Compare access with daily work interval. |
| `badge_access_log` | `video_cameras` | Indirect room | `badge_access_log.room_id = video_cameras.room_id` | Find cameras covering badge-event room. |
| `badge_access_log` | `video_footage` | Direct PK/FK (optional) | `video_footage.badge_access_id = badge_access_log.badge_access_id` | Optional event-footage link. |
| `badge_access_rules` | `departments` | Indirect via employees | `badge_access_rules.employee_id = employees.employee_id AND employees.department_id = departments.department_id` | Analyze permissions by department. |
| `badge_access_rules` | `employee_status_history` | Temporal employee | `badge_access_rules.employee_id = employee_status_history.employee_id; compare effective_date to rule validity` | Detect permissions persisting through status changes. |
| `badge_access_rules` | `employees` | Direct PK/FK | `badge_access_rules.employee_id = employees.employee_id` | Rules assigned to employee. |
| `badge_access_rules` | `job_titles` | Indirect via employees | `badge_access_rules.employee_id = employees.employee_id AND employees.job_title_id = job_titles.job_title_id` | Analyze permissions by job role. |
| `badge_access_rules` | `key_access_log` | Logical authorization | `rule room = key_inventory.room_id via key_access_log.key_id; participant employee = rule employee; checkout date within validity` | Validate key custodian authorization. |
| `badge_access_rules` | `key_inventory` | Room domain | `badge_access_rules.room_id = key_inventory.room_id` | Find key for authorized room. |
| `badge_access_rules` | `rooms` | Direct PK/FK | `badge_access_rules.room_id = rooms.room_id` | Rules assigned to room. |
| `badge_access_rules` | `time_clock` | Indirect employee | `badge_access_rules.employee_id = time_clock.employee_id` | Compare authorization holders with work activity. |
| `badge_access_rules` | `time_clock_summary` | Indirect employee | `badge_access_rules.employee_id = time_clock_summary.employee_id` | Compare authorization holders with daily work summaries. |
| `badge_access_rules` | `video_cameras` | Room domain | `badge_access_rules.room_id = video_cameras.room_id` | Find cameras for authorized rooms. |
| `badge_access_rules` | `video_footage` | Indirect room/event | `rule room -> video_cameras.room_id -> video_footage.camera_id, or through badge_access_log` | Find video tied to governed rooms/events. |
| `departments` | `employee_status_history` | Indirect via employees | `departments.department_id = employees.department_id AND employee_status_history.employee_id = employees.employee_id` | Status history by department. |
| `departments` | `employees` | Direct PK/FK | `employees.department_id = departments.department_id` | Employee department assignment. |
| `departments` | `job_titles` | Domain/semantic | `departments.department_name ≈ job_titles.job_category` | Partial organizational-domain overlap only; do not treat as FK. |
| `departments` | `key_access_log` | Indirect via employees | `departments.department_id = employees.department_id AND key participant = employees.employee_id` | Key activity by department. |
| `departments` | `key_inventory` | Domain via building | `departments.location = rooms.building AND key_inventory.room_id = rooms.room_id` | Keys in department building; not authorization. |
| `departments` | `rooms` | Domain location | `departments.location = rooms.building` | Shared building/location domain; not FK. |
| `departments` | `time_clock` | Employee + domain | `departments.department_id = employees.department_id AND time_clock.employee_id = employees.employee_id; optionally departments.location = time_clock.location` | Compare employee punch location with home department location. |
| `departments` | `time_clock_summary` | Indirect via employees | `departments.department_id = employees.department_id AND time_clock_summary.employee_id = employees.employee_id` | Daily summaries by department. |
| `departments` | `video_cameras` | Domain via room | `departments.location = rooms.building AND video_cameras.room_id = rooms.room_id` | Cameras in department building. |
| `departments` | `video_footage` | Indirect domain | `departments.location = rooms.building -> video_cameras -> video_footage` | Footage in department building. |
| `employee_status_history` | `employees` | Direct PK/FK | `employee_status_history.employee_id = employees.employee_id` | Status history for employee. |
| `employee_status_history` | `job_titles` | Indirect via employees | `employee_status_history.employee_id = employees.employee_id AND employees.job_title_id = job_titles.job_title_id` | Status history by role. |
| `employee_status_history` | `key_access_log` | Temporal employee | `key participant = employee_status_history.employee_id AND effective_date <= DATE(checked_out_at)` | Employment status at key checkout. |
| `employee_status_history` | `key_inventory` | Indirect through key log | `status employee -> key_access_log participant -> key_inventory.key_id` | Keys used during a status period. |
| `employee_status_history` | `rooms` | Indirect through access/rules | `status employee -> badge/key/rules -> rooms` | Rooms associated with employees by status period. |
| `employee_status_history` | `time_clock` | Temporal employee | `employee_status_history.employee_id = time_clock.employee_id AND effective_date <= DATE(punch_time)` | Clock activity vs employment status. |
| `employee_status_history` | `time_clock_summary` | Temporal employee | `employee_status_history.employee_id = time_clock_summary.employee_id AND effective_date <= work_date` | Daily work vs employment status. |
| `employee_status_history` | `video_cameras` | Indirect | `status employee -> badge event room -> video_cameras.room_id` | Cameras covering rooms accessed during status period. |
| `employee_status_history` | `video_footage` | Indirect temporal | `status employee -> badge_access_log -> video_footage.badge_access_id` | Footage for employee access during status period. |
| `employees` | `job_titles` | Direct PK/FK | `employees.job_title_id = job_titles.job_title_id` | Employee job-title assignment. |
| `employees` | `key_access_log` | Direct PK/FK | `key_access_log.primary_employee_id = employees.employee_id OR key_access_log.secondary_employee_id = employees.employee_id` | Primary and optional dual-custody participant. |
| `employees` | `key_inventory` | Indirect | `employee -> key_access_log participant -> key_inventory.key_id OR employee -> access rules -> room -> key_inventory` | Keys used by or relevant to employee. |
| `employees` | `rooms` | Indirect | `employee -> badge_access_rules/badge_access_log/key_access_log -> rooms` | Rooms authorized for or accessed by employee. |
| `employees` | `time_clock` | Direct PK/FK | `time_clock.employee_id = employees.employee_id` | Raw clock activity. |
| `employees` | `time_clock_summary` | Direct PK/FK | `time_clock_summary.employee_id = employees.employee_id` | Daily time summary. |
| `employees` | `video_cameras` | Indirect | `employee -> badge/rules room -> video_cameras.room_id` | Cameras covering employee-associated rooms. |
| `employees` | `video_footage` | Indirect | `employee -> badge_access_log -> video_footage.badge_access_id` | Footage tied to employee badge events. |
| `job_titles` | `key_access_log` | Indirect via employees | `job_titles.job_title_id = employees.job_title_id AND key participant = employees.employee_id` | Key activity by role. |
| `job_titles` | `key_inventory` | Indirect | `job_titles -> employees -> key_access_log -> key_inventory` | Keys used by role. |
| `job_titles` | `rooms` | Indirect | `job_titles -> employees -> badge rules/events -> rooms` | Rooms associated with role. |
| `job_titles` | `time_clock` | Indirect | `job_titles -> employees -> time_clock` | Clock activity by role. |
| `job_titles` | `time_clock_summary` | Indirect | `job_titles -> employees -> time_clock_summary` | Daily work by role. |
| `job_titles` | `video_cameras` | Indirect | `job_titles -> employees -> access room -> video_cameras` | Camera coverage of role-associated rooms. |
| `job_titles` | `video_footage` | Indirect | `job_titles -> employees -> badge_access_log -> video_footage` | Footage tied to role activity. |
| `key_access_log` | `key_inventory` | Direct PK/FK | `key_access_log.key_id = key_inventory.key_id` | Key event to physical key. |
| `key_access_log` | `rooms` | Indirect via key | `key_access_log.key_id = key_inventory.key_id AND key_inventory.room_id = rooms.room_id` | Resolve key event to room. |
| `key_access_log` | `time_clock` | Temporal employee | `key participant = time_clock.employee_id; compare checkout/return with punch interval` | Check key use while on duty. |
| `key_access_log` | `time_clock_summary` | Temporal employee/day | `key participant = time_clock_summary.employee_id AND DATE(checked_out_at)=work_date` | Compare key use with daily work interval. |
| `key_access_log` | `video_cameras` | Indirect room | `key_access_log.key_id -> key_inventory.room_id = video_cameras.room_id` | Find cameras covering key-associated room. |
| `key_access_log` | `video_footage` | Indirect room + time | `key -> room -> camera -> footage; compare footage interval with checkout/return interval` | Corroborate key activity with video. |
| `key_inventory` | `rooms` | Direct PK/FK (1:1 by UNIQUE) | `key_inventory.room_id = rooms.room_id` | Key to room; UNIQUE room_id means at most one key row per room. |
| `key_inventory` | `time_clock` | Domain location | `key_inventory.room_id -> rooms.building = time_clock.location` | Clock activity in key room building. |
| `key_inventory` | `time_clock_summary` | Indirect through key log | `key_inventory -> key_access_log participant/date -> time_clock_summary employee/work_date` | Daily work of key custodians. |
| `key_inventory` | `video_cameras` | Room domain | `key_inventory.room_id = video_cameras.room_id` | Cameras for key-associated room. |
| `key_inventory` | `video_footage` | Indirect room/camera | `key_inventory.room_id = video_cameras.room_id AND video_footage.camera_id = video_cameras.camera_id` | Footage for key-associated room. |
| `rooms` | `time_clock` | Domain location | `rooms.building = time_clock.location` | Shared building/location domain; not FK. |
| `rooms` | `time_clock_summary` | Indirect | `room -> badge/key/rules employee -> time_clock_summary.employee_id` | Work summaries of people associated with room. |
| `rooms` | `video_cameras` | Direct PK/FK | `video_cameras.room_id = rooms.room_id` | Camera coverage by room. |
| `rooms` | `video_footage` | Indirect via camera | `rooms.room_id = video_cameras.room_id AND video_footage.camera_id = video_cameras.camera_id` | All footage for a room. |
| `time_clock` | `time_clock_summary` | Logical derivation | `time_clock.employee_id = time_clock_summary.employee_id AND DATE(punch_time)=work_date` | Raw punches to daily summary. |
| `time_clock` | `video_cameras` | Domain via building | `time_clock.location = rooms.building AND video_cameras.room_id = rooms.room_id` | Cameras at punch building. |
| `time_clock` | `video_footage` | Domain + temporal | `time_clock.location = rooms.building -> cameras -> footage; compare punch_time to footage interval` | Footage near punch location/time. |
| `time_clock_summary` | `video_cameras` | Indirect | `summary employee -> badge/key room -> video_cameras.room_id` | Cameras for employee activity during workday. |
| `time_clock_summary` | `video_footage` | Indirect temporal | `summary employee/date -> badge event -> video_footage` | Footage associated with workday access. |
| `video_cameras` | `video_footage` | Direct PK/FK | `video_footage.camera_id = video_cameras.camera_id` | Footage to recording camera. |

## Important interpretation notes

- `MUL` is index metadata, not proof of a declared foreign key.
- `key_inventory.room_id` is `UNI`, so the schema enforces at most one key-inventory row per room.
- Domain joins are explicitly separated from direct key joins.
- Temporal joins should use the state effective at the event timestamp, not merely the latest row overall.
- `employee_status_history.changed_by` is free text and should not be treated as an employee FK.
- `badge_access_rules.valid_until` is inclusive.
- `secondary_employee_id` represents the second participant in dual-custody scenarios.
- `Tailgate` represents a detected tailgating security incident.
- `time_clock_summary.hours_worked` is elapsed time with no meal/break deduction.
