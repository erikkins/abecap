---
name: Social post images live in S3 price-data bucket, NOT frontend bucket
description: Social post images must be updated in s3://rigacap-prod-price-data-149218244179/social/images/ — the frontend CDN path is only for display
type: feedback
---

Social post images are stored in TWO places:
1. **Frontend CDN:** `s3://rigacap-prod-frontend-149218244179/launch-cards/` → shown on the website
2. **Social publisher:** `s3://rigacap-prod-price-data-149218244179/social/images/` → uploaded to Twitter/Instagram when publishing

**When regenerating launch card PNGs, update BOTH locations.** The social publisher reads from the price-data bucket, NOT the frontend bucket.

**Why:** On Apr 8 2026, we regenerated the launch card PNGs and updated the frontend bucket, but the social publisher kept uploading the old images from the price-data bucket. Had to delete and re-post 3 times before finding the issue. Twitter/Instagram then blocked re-posting due to duplicate content spam filters.

**How to apply:**
1. Regenerate PNGs → `frontend/public/launch-cards/`
2. Copy to S3 frontend bucket (CI/CD does this)
3. **ALSO copy to:** `aws s3 cp frontend/public/launch-cards/launch-N.png s3://rigacap-prod-price-data-149218244179/social/images/launch-N.png --profile rigacap`
4. Verify BEFORE publishing: check the S3 social images bucket, not just the CDN
5. **Never re-post the same text to Twitter/Instagram more than once** — spam filters will block it
