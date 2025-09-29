#!/usr/bin/env python3
"""
Scheduler stub: prints gcloud commands for Cloud Run Jobs / Cloud Scheduler.
No live mutations; set DRY_RUN=0 to actually run gcloud if available.
"""
import os


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT', '<project>')
    region = os.getenv('AELP2_REGION', 'us-central1')
    job = 'aelp2-nightly'
    cmd = f"gcloud run jobs create {job} --image gcr.io/{project}/aelp2:latest --region {region} --args='python3 -m AELP2.ops.prefect_flows'"
    sched = f"gcloud scheduler jobs create http {job}-sched --schedule='0 6 * * *' --uri='https://run.googleapis.com/apis/run.googleapis.com/v1/namespaces/{project}/jobs/{job}:run' --http-method=POST --oauth-service-account-email=<sa>@{project}.iam.gserviceaccount.com"
    print('[dry_run] Commands to create job + schedule:')
    print(cmd)
    print(sched)


if __name__ == '__main__':
    main()

