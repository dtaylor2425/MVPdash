# Options flow editorial publication contract

This release changes operational reads and adds an editorial archive. It does not change
the frozen activity formula, committed research reports, or historical research loader default.

## API

All routes retain the existing internal-token dependency. The Next.js gateway must still
authorize its private allow-list and never expose the service credential.

- `GET /api/private/options-flow/latest?session=YYYY-MM-DD`: exact session, full-flow only;
  absent date selects newest available full-flow session. No older ticker carry-forward.
- `GET /ticker/{ticker}?session=YYYY-MM-DD&include_trades=true`: same session and precedence,
  returns `{ticker: {...}, sessionDate}`. Intraday and large trades only appear in detail.
- `GET /history?ticker=SPY&days=20&session=YYYY-MM-DD`: optional inclusive upper date bound.
- `GET /sessions`: `{sessions:[{sessionDate}]}` descending, at most260 sessions.
- `GET /status`: most recently published and latest live collection attempt separately.

Latest adds `availableSessions`, `session`, `collectionStatus`, and `dailyBrief`.
Session includes `sessionDate`, `state` (`historical`, `final`, `partial`), `comparable`,
coverage, exact `snapshotIds`, and `evidenceSha256`. Each ticker has `provenance`
with snapshot/run IDs, source, mode, methodology, collection profile, state and comparability.
Quality is passed through unchanged: final means completed collection window, not perfect data
or certainty about the investor's intent. Historical state is distinct from live final state.
Legacy live rows lacking completion metadata are partial and excluded from daily baselines.

Daily comparisons use the immediately previous exchange trading session; they never skip gaps
to manufacture a daily change. Partial/missing/different-methodology or different-collection-profile
symbol pairs have null deltas and an explicit reason. IV/skew/term spread and zero-DTE shares
are fractions; multiply deltas by100 for percentage points. Spot is a stored underlying Greek
snapshot reference, not an official equity close. Activity persistence is descriptive,
counts consecutive eligible sessions with historical percentile>=75 and resets on missing dates.

Production activity includes historical full-flow plus explicitly final live full-flow.
Operational baselines restart after any missing exchange session, methodology change, unknown
collection profile, or change in session bounds/DTE/strike range/Greek sampling settings. The
unchanged prior-20/minimum-10 formula runs within each compatible segment; it is suppressed
until enough consistent observations accumulate. Percentiles and persistence use the same
compatible collection identities and available ETF set. Frozen research computation stays unchanged.
Final observations precede partial observations; final live wins over historical for the same date.
An exact-date activity factor is required; an old factor is never stamped with a newer session.
Operational derived cache is bounded and keyed by all eligible publication IDs, selected date
and chart length. Reruns/corrections must publish new snapshot IDs; in-place premium corrections
are not supported. Frozen issue evidence is copied separately and survives raw snapshot pruning.

## Frozen issue revisions

Apply `sql/011_options_flow_issues.sql` explicitly before enabling archive controls. This is
an additive table/index/trigger migration and does not touch vendor or research data. No runtime
API DDL. Updates/deletes on revisions are rejected by the database trigger.

`POST /issues` accepts `{session,title,body,chartSpec,snapshotIds,expectedEvidenceSha256,
correctionOf?,correctionReason?}`. IDs are UUIDs; expected hash is64 lowercase hexadecimal chars.
The server reads a repeatable-read analytical snapshot, requires both exact current IDs and the
reviewed evidence hash, and stores a separate copy plusSHA256. Conflict409 means refresh/review.
Corrections insert a new UUID and require an existing same-session revision plus reason.
`chartSpec` is editorial JSON and may contain chart SVG/version/caption; it is not trusted as
the numerical evidence. Summary evidence excludes raw prints, intraday detail and operational
diagnostics. GET `/issues?session=...` lists metadata; GET `/issues/{id}` retrieves the stored revision.
Archive requests fail503 until migration is applied. No Substack posting or automatic distribution.

## Deployment and operational acceptance

1. Apply migration011 to the intended database with backup/transaction and verify table+trigger.
2. Deploy API and worker from the same reviewed source; deploy frontend gateway/UI contract.
3. Configure actual Railway worker service settings directly; this environment rejects the
   deprecated config-as-code file attachment. `railway.theta-worker.json` is a reviewed reference,
   not a file to attach. Remove any dashboard `--force` override. Cron is `45 21,22 * * 1-5` UTC: two daily attempts,
   both safely after16:30 Eastern in winter and summer. The first successful complete-universe
   final run makes the second a no-op. Partial/universe-subset runs remain retryable.
4. Live collector holds the existing backfill Postgres session advisory lock until connection close,
   preventing overlapping live/backfill vendor sessions even across commits. Configure provider/container timeouts and external
   alerting for exit1/2, absent final publication, missing tickers and excessive duration.
5. Verify an actual completed final collection, exact-date detail, new eligible activity session,
   coverage and last-attempt status. A deployed service or cron field alone is not acceptance.
6. Freeze an issue in a test environment, publish an analytical correction and verify old issue
   hash/evidence remains unchanged and stale-reviewed exports return409. Validate database
   immutability and repeated scheduler attempts against a staging database before production.

Local regression tests use synthetic fixtures and mocked DB/provider calls. SQL bindings,
selection precedence, archive migration, corrections and UPDATE/DELETE rejection were also tested
in an isolated PostgreSQL schema inside one rolled-back transaction; the schema was verified absent
afterward. Read-only production research parity tests passed.

Deployment coordinator confirmed on September 25: migration011 applied transactionally,
empty archive and immutability trigger verified; API and worker universe explicitly set to
SPY, QQQ, IWM, SMH, TLT, GLD; actual worker schedule set to the two UTC attempts above,
start command without force, restart policy NEVER, and Dockerfile.theta-worker selected via
service settings and RAILWAY_DOCKERFILE_PATH. Deprecated railwayConfigFile attachment is null.
Application deployment and successful new vendor collection/actual recurrence still require
separate operational verification; configuration alone is not a successful collection.
