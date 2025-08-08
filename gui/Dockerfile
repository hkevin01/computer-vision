# Multi-stage Dockerfile for Stereo Vision Web GUI

# === Development Stage ===
FROM python:3.9-alpine as development
WORKDIR /app
COPY . .
EXPOSE 3000
CMD ["python", "-m", "http.server", "3000"]

# === Production Stage ===
FROM nginx:alpine as production

# Install dependencies for health checks
RUN apk add --no-cache curl wget

# Remove default nginx website
RUN rm -rf /usr/share/nginx/html/*

# Copy our web application
COPY . /usr/share/nginx/html/

# Create custom nginx configuration
COPY <<EOF /etc/nginx/conf.d/default.conf
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # Enable gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    # API proxy (if needed for CORS)
    location /api/ {
        proxy_pass http://api:8080/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Serve static files
    location / {
        try_files \$uri \$uri/ /index.html;
        add_header Cache-Control "no-cache, must-revalidate";
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\\n";
        add_header Content-Type text/plain;
    }
}
EOF

# Create startup script
COPY <<EOF /docker-entrypoint.d/10-configure-api.sh
#!/bin/sh
# Replace API_URL placeholder in JavaScript files
if [ ! -z "\$API_URL" ]; then
    echo "Configuring API URL: \$API_URL"
    # Update config.json with the actual API URL
    if [ -f /usr/share/nginx/html/config.json ]; then
        sed -i "s|http://localhost:8080|\$API_URL|g" /usr/share/nginx/html/config.json
    fi
    # Update app.js if it contains hardcoded URLs
    if [ -f /usr/share/nginx/html/app.js ]; then
        sed -i "s|http://localhost:8080|\$API_URL|g" /usr/share/nginx/html/app.js
    fi
fi
EOF

RUN chmod +x /docker-entrypoint.d/10-configure-api.sh

EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
