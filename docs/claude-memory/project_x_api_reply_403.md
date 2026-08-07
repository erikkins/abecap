---
name: project_x_api_reply_403
description: X/Twitter Free API tier CANNOT auto-post replies to 3rd-party tweets (403) — we use a deep-link workaround
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

**X API Free tier blocks programmatic replies to third-party tweets.** Publishing an approved contextual reply returns `403 not-authorized-for-resource: "You can only reply to or quote posts where you are mentioned or are the author."` This is an X API ACCESS-TIER restriction, NOT a bug — our `POST /2/tweets` reply plumbing (social_posting_service.post_to_twitter, `payload["reply"]={"in_reply_to_tweet_id":...}`, OAuth 1.0a user context) is correct. This is why Erik had been copy-pasting replies manually.

**Why:** X restricts automated replies to non-mention/non-author tweets on the Free tier to fight spam. Basic tier (~$200/mo) *might* lift it but is UNVERIFIED — do not assume paying fixes it (X has blocked automated 3rd-party replies even on paid tiers). Verify before recommending a spend.

**How to apply:** The reply-scanner (`reply_scanner_service.py`) drafts contextual replies → the approval email button is a DEEP LINK (not API auto-post): `GET /api/admin/social/posts/{id}/compose-email?token=` → marks the draft handled → 302 to X Web Intent `https://twitter.com/intent/tweet?in_reply_to={tweet_id}&text={urlencoded}` which opens the X composer as a reply, text pre-filled; Erik taps Post. $0, no API write, no 403, no copy-paste. The auto-publish cron (`post_scheduler_service._publish_scheduled_posts`) and `auto_schedule_drafts` both EXCLUDE `post_type='contextual_reply'` so automation never re-triggers the 403. Our OWN original posts (not replies) still API-publish fine — the restriction is only replies to others. See [[session-progress]].
