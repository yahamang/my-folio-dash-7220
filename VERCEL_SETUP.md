# Vercel Setup Guide - Dashboard Refresh Button

This guide explains how to deploy the refresh button serverless function to Vercel.

## Overview

The refresh button in the dashboard triggers a GitHub Actions workflow to regenerate the dashboard with the latest data. To protect your GitHub Personal Access Token (PAT) from being exposed in the static HTML, we use a Vercel Edge Function as a secure proxy.

## Prerequisites

- GitHub Personal Access Token with `repo` scope (for triggering Actions)
- Vercel account (free tier is sufficient)

## Step 1: Create GitHub Personal Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Set a descriptive name: "Investment Dashboard Refresh"
4. Select scope: **`repo`** (Full control of private repositories)
5. Click "Generate token"
6. **IMPORTANT**: Copy the token immediately (e.g., `ghp_xxxxxxxxxxxx`)

## Step 2: Install Vercel CLI

```bash
npm install -g vercel
```

## Step 3: Link Your Project to Vercel

```bash
cd ~/investment
vercel link
```

Follow the prompts:
- Set up and deploy? → Yes
- Which scope? → Select your account
- Link to existing project? → No
- Project name? → Press Enter (use default: `investment`)
- Directory? → Press Enter (use default: `./`)

## Step 4: Add GitHub PAT as Vercel Secret

```bash
vercel env add GITHUB_PAT
```

When prompted:
1. What's the value of GITHUB_PAT? → Paste your GitHub PAT (from Step 1)
2. Add GITHUB_PAT to which Environments? → Select **Production** and **Preview** (use spacebar to select)
3. Press Enter to confirm

## Step 5: Deploy to Vercel

```bash
vercel --prod
```

This will:
1. Build and deploy your project
2. Output a production URL (e.g., `https://investment-abc123.vercel.app`)
3. Make the serverless function available at that URL

## Step 6: Update Dashboard with Vercel URL

After deployment, update the JavaScript function in `dashboard.py`:

1. Open `dashboard.py`
2. Find the line with `'https://YOUR_VERCEL_URL/api/trigger-refresh'`
3. Replace `YOUR_VERCEL_URL` with your actual Vercel production URL
4. Example: `'https://investment-abc123.vercel.app/api/trigger-refresh'`
5. Regenerate the dashboard: `python3 dashboard.py`
6. Commit and push the changes

## Step 7: Test the Refresh Button

1. Open `dashboard.html` in your browser
2. Click the "새로고침" (Refresh) button in the header
3. It should show "업데이트 중..." with a spinning icon
4. After 2-3 seconds, it should show "✓ 시작됨" (Started)
5. Check your GitHub repository's Actions tab to verify the workflow was triggered

## Troubleshooting

### Button shows "✗ 실패" (Failed)

1. Check browser console for errors (F12 → Console)
2. Verify your Vercel URL is correct in `dashboard.py`
3. Check that the serverless function is deployed: Visit `https://YOUR_VERCEL_URL/api/trigger-refresh` (should show "Method not allowed")

### GitHub workflow not triggering

1. Verify your GitHub PAT has `repo` scope
2. Check Vercel function logs: `vercel logs https://YOUR_VERCEL_URL/api/trigger-refresh`
3. Ensure the repository name in `api/trigger-refresh.js` is correct: `yahamang/my-folio-dash-7220`

### Environment variable not found

```bash
# Re-add the secret
vercel env add GITHUB_PAT

# Force redeployment
vercel --prod --force
```

## Security Notes

1. **Never commit your GitHub PAT to the repository**
2. The PAT is stored securely in Vercel's environment variables
3. The Edge Function only exposes a safe API endpoint
4. Consider adding IP restrictions to your GitHub PAT for extra security

## Updating the Function

If you need to modify the serverless function:

```bash
# Edit api/trigger-refresh.js
# Then redeploy
vercel --prod
```

## Cost

- Vercel Free Tier includes:
  - 100 GB bandwidth per month
  - 100,000 Edge Function invocations per month
  - More than sufficient for personal dashboard use

---

## Alternative: Direct GitHub API (Not Recommended)

If you don't want to use Vercel, you can call the GitHub API directly from the browser, but this exposes your PAT in the HTML source code. Only do this if your repository is **private** and you accept the security risk.

See the plan file for details on this approach.
