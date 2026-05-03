---
name: Close direct execute-api WAF bypass with shared-secret header
description: After May 3 2026 CloudFront cutover, api.rigacap.com is WAF-protected, but the underlying execute-api URL is still publicly reachable. ~30 min hardening pass to inject a shared secret header at CloudFront and reject anything without it at FastAPI middleware.
type: project
originSessionId: 1081317d-f863-470f-ab07-65ddc614e856
---
**Why this exists:** May 3 2026 CloudFront cutover put WAF in front of `api.rigacap.com` (the new CloudFront distribution `E261YQC1EOD078` fronts API Gateway HTTP API v2). All subscriber-facing surfaces — frontend, mobile app, Stripe webhooks — go through `api.rigacap.com` and are WAF-protected.

**The hole:** the underlying API Gateway execute-api URL (`https://0f8to4k21c.execute-api.us-east-1.amazonaws.com`) is publicly reachable. An attacker who discovers it can hit the API directly, bypassing CloudFront and WAF entirely. The URL is somewhat obscured (not advertised in any documentation) but findable in CloudFormation/Terraform outputs, DNS history, AWS console screenshots, etc.

**The fix (~30 min):**

1. **Generate a shared secret** — random ~32-char string. Store in AWS Secrets Manager OR as a Terraform variable in the deploy pipeline.
2. **CloudFront origin custom header** — add a custom header on the API distribution origin block:
   ```hcl
   origin {
     ...
     custom_header {
       name  = "X-Origin-Verify"
       value = var.cloudfront_origin_secret
     }
   }
   ```
3. **FastAPI middleware** — reject any request that doesn't carry the matching header. New file `backend/app/core/origin_guard.py`:
   ```python
   from fastapi import Request
   from starlette.middleware.base import BaseHTTPMiddleware
   from starlette.responses import PlainTextResponse
   import os, hmac

   class OriginVerifyMiddleware(BaseHTTPMiddleware):
       async def dispatch(self, request, call_next):
           # Allow OPTIONS preflight + health check unconditionally
           if request.method == "OPTIONS" or request.url.path == "/health":
               return await call_next(request)
           expected = os.environ.get("CLOUDFRONT_ORIGIN_SECRET", "")
           if not expected:  # not configured = guard disabled
               return await call_next(request)
           supplied = request.headers.get("X-Origin-Verify", "")
           if not hmac.compare_digest(expected, supplied):
               return PlainTextResponse("Forbidden", status_code=403)
           return await call_next(request)
   ```
4. **Wire the env var** — add `CLOUDFRONT_ORIGIN_SECRET` to API Lambda env vars in Terraform (alongside `TRUST_FORWARDED_FOR`).
5. **Deploy in lockstep** — Terraform apply must update CloudFront (origin header) AND API Lambda (env var) atomically. Otherwise either:
   - CloudFront sends header but Lambda doesn't check → no harm
   - CloudFront doesn't send header but Lambda checks → all traffic blocked. **Avoid this order.**
6. **Validate** — after apply: `curl https://api.rigacap.com/api/market-data-status` should still 200 (header injected by CloudFront), `curl https://0f8to4k21c.execute-api.us-east-1.amazonaws.com/api/market-data-status` should now 403 (direct hit, no header).

**Priority:** Low-medium. The execute-api URL isn't advertised; no known attack ongoing. But standard hardening practice and worth doing once before any external attention. Schedule alongside CB-in-production wiring or whenever a Terraform window opens.

**Connected:**
- May 3 CloudFront cutover (this whole effort)
- TRUST_FORWARDED_FOR plumbing — done; same env-var deploy pattern as the secret would use
