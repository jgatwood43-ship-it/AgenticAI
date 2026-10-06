# Keys Access Log

**Database table:** `keys_access_log`

## What this table is for

Tracks physical key checkout/access sessions, including custodians, checkout/return timestamps, purpose, and status.

## At a glance

- **Primary key:** `access_id`
- **Table role:** Transaction / history data
- **Data classification:** Anonymous test data

## Column guide

| Column | Inferred Type | Nullable | Key / Reference | Description |
|---|---|---:|---|---|
| `access_id` | INTEGER | No | PK | Unique identifier for the key-access/checkout record. |
| `key_id` | INTEGER | No | FK → `key_inventory.key_id` | Physical key used or checked out. |
| `primary_employee_id` | INTEGER | No | FK → `employees.employee_id` | Primary employee responsible for the key/access session. |
| `secondary_employee_id` | INTEGER | Yes | FK → `employees.employee_id` | Second employee participating in the key/access session when dual custody is required. |
| `checked_out_at` | TIMESTAMP | No | — | Timestamp when the key was checked out or the keyed access session began. |
| `returned_at` | TIMESTAMP | Yes | — | Timestamp when the key was returned; NULL may indicate it has not been returned. |
| `purpose` | VARCHAR/TEXT | No | — | Business reason documented for the key access. |
| `status` | VARCHAR/TEXT | No | — | Current/final state of the key-access record. Observed values: Returned and Overdue. |

## How this table relates to other tables

| From / Basis | To | Cardinality / Use | Confidence | Description |
|---|---|---|---|---|
| `keys_access_log.key_id` | `key_inventory.key_id` | Many-to-one | Strong / direct ID match | Each access record uses one physical key. |
| `keys_access_log.primary_employee_id` | `employees.employee_id` | Many-to-one | Strong / direct ID match | Primary employee responsible for the key. |
| `keys_access_log.secondary_employee_id` | `employees.employee_id` | Many-to-one, optional | Strong / direct ID match | Secondary employee participating in the dual-custody key-access session. |
| `keys_access_log.key_id → key_inventory.room_id` | `rooms.room_id` | Indirect many-to-one | Strong indirect relationship | The key determines the room being accessed. |
| `keys_access_log ↔ badge_access_log` | room + employees + timestamps | Event/session correlation | Strong logical relationship | Badge IN/OUT events closely bracket checkout/return times in the supplied data. |
| `keys_access_log ↔ badge_access_rules` | employee + key room + time | Authorization validation | Strong logical relationship | Employee's authorization can be checked for the key's room. |

## Business rules and data notes

- access_id is distinct from badge_access_log.badge_access_id and should not be joined to it by numeric equality.
- Record 25 is Overdue with `returned_at` NULL and corresponds to a badge IN event for employee 8 in room 9 without a supplied OUT event. This is retained as an intentional anomaly in the test dataset.

## Recommended database safeguards

- Primary key on `access_id`.
- Foreign key candidate: `key_id` → `key_inventory.key_id`.
- Foreign key candidate: `primary_employee_id` → `employees.employee_id`.
- Foreign key candidate: `secondary_employee_id` → `employees.employee_id`.
- Index `(primary_employee_id, checked_out_at)` and `key_id`.
- Consider CHECK logic requiring `returned_at >= checked_out_at` when returned_at is not NULL.

## Useful security-analysis questions

- Find overdue or unreturned keys.
- Validate dual-custody key use.
- Compare key use with badge activity and room authorization.

## Test-data reminder

This table contains **anonymous test data**. Some records are intentionally suspicious, incomplete, or inconsistent so that security analytics, anomaly detection, and agent reasoning can be tested. Do not automatically “correct” an anomaly unless the test scenario explicitly calls for it.
