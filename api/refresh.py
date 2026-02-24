from http.server import BaseHTTPRequestHandler
import sys
import os

# Add parent directory to path to import dashboard
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)

# Change working directory to parent (where config.json is)
os.chdir(parent_dir)

from dashboard import generate_dashboard_html

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Generate fresh HTML with latest data
            # Skip Binance to stay under 10s Vercel limit
            html = generate_dashboard_html(skip_binance=True)

            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))

        except Exception as e:
            # Error handling
            self.send_response(500)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            error_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Error</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .error-box {{
            background: white;
            border: 2px solid #ef4444;
            border-radius: 8px;
            padding: 20px;
        }}
        h1 {{
            color: #ef4444;
            margin-top: 0;
        }}
        pre {{
            background: #fee;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 12px;
        }}
        .actions {{
            margin-top: 20px;
        }}
        button {{
            background: #3b82f6;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{
            background: #2563eb;
        }}
    </style>
</head>
<body>
    <div class="error-box">
        <h1>⚠️ Dashboard Generation Failed</h1>
        <p>서버에서 대시보드를 생성하는 중 오류가 발생했습니다.</p>
        <pre>{str(e)}</pre>
        <div class="actions">
            <button onclick="location.reload()">다시 시도</button>
            <button onclick="window.history.back()">돌아가기</button>
        </div>
    </div>
</body>
</html>"""
            self.wfile.write(error_html.encode('utf-8'))

    def do_POST(self):
        # Support POST for consistency with old API
        self.do_GET()
