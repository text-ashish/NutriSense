# Build stage
FROM node:22 AS builder

WORKDIR /app

COPY frontend/package*.json ./
RUN npm install

COPY frontend/. .
RUN npm run build

# Serve stage
FROM nginx:1.25

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
