# Phase 0 — account hardening (do once, before any build)

Click-only work in the AWS Console. This is what makes the ~$100 credit budget safe. Do the cost guardrails **first** — they are the safety net for everything after.

## Cost guardrails (do these first)

- [ ] **AWS Budgets** — a monthly cost budget with email alerts at **$10 / $25 / $50 / $75**.
- [ ] **Cost Explorer** enabled (daily spend by service).
- [ ] **CloudWatch billing alerts** turned on.
- [ ] **Cost-allocation tag** `project` activated (Billing → Cost allocation tags) — every resource this track creates is tagged `project=demand-forecasting`, so you can filter spend and find anything left running.

## Identity & access (stop using root)

- [ ] **MFA on the root user**; then stop logging in as root.
- [ ] A daily **IAM user (or IAM Identity Center user) with MFA** for all work.
- [ ] That user has access to **S3, Athena, Glue** for Phase 1 (an admin-ish policy is fine to start; tighten to least-privilege later — itself an exam topic).
- [ ] **Access key pair** created for that IAM user, for the CLI. ⚠️ The keys go **only** into `aws configure` (`~/.aws/credentials`) — never into the repo or any committed file.

## Region & credits

- [ ] Region locked to **us-east-1** (cheapest, has every service; working across regions creates orphaned billable resources).
- [ ] Billing → Credits: confirm the credits **cover S3, Athena, Glue, and SageMaker**.

## Naming

- [ ] Choose a **globally-unique S3 bucket name**, e.g. `demand-forecasting-<your-initials>-<a-few-digits>` (S3 names are global). You pass this to `s3_setup.py`.

## End-of-session habit

Every time you stop working: check Cost Explorer, and confirm no endpoint, no NAT Gateway, no running SageMaker Studio app, and no stuck job. Spending should read a few dollars at most.
