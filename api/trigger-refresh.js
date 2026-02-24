// Vercel Edge Function to trigger GitHub Actions workflow
// This proxy protects the GitHub PAT from being exposed in static HTML

export const config = { runtime: 'edge' };

export default async function handler(req) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return new Response(
      JSON.stringify({ error: 'Method not allowed' }),
      {
        status: 405,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }

  // Get GitHub PAT from environment variable
  const GITHUB_PAT = process.env.GITHUB_PAT;
  if (!GITHUB_PAT) {
    return new Response(
      JSON.stringify({ error: 'GitHub PAT not configured' }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }

  try {
    // Trigger GitHub Actions workflow
    const response = await fetch(
      'https://api.github.com/repos/yahamang/my-folio-dash-7220/actions/workflows/dashboard.yml/dispatches',
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${GITHUB_PAT}`,
          'Accept': 'application/vnd.github.v3+json',
          'Content-Type': 'application/json',
          'User-Agent': 'Investment-Dashboard-Refresh'
        },
        body: JSON.stringify({ ref: 'main' })
      }
    );

    // GitHub API returns 204 No Content on success
    if (response.status === 204) {
      return new Response(
        JSON.stringify({
          success: true,
          message: 'Workflow triggered successfully'
        }),
        {
          status: 200,
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
          }
        }
      );
    }

    // Handle non-204 responses
    const errorText = await response.text();
    return new Response(
      JSON.stringify({
        error: 'Failed to trigger workflow',
        status: response.status,
        details: errorText
      }),
      {
        status: response.status,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        }
      }
    );
  } catch (error) {
    return new Response(
      JSON.stringify({
        error: 'Internal server error',
        message: error.message
      }),
      {
        status: 500,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        }
      }
    );
  }
}
