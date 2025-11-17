FROM node:18

WORKDIR /app

# Copy only backend package files
COPY backend/package*.json ./

RUN npm install

# Copy backend source code
COPY backend/. .

EXPOSE 5000

CMD ["npm", "start"]
