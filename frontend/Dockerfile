FROM node:18

WORKDIR /app

# copy ONLY the frontend package.json files
COPY frontend/package*.json ./

RUN npm install

# copy the entire frontend folder
COPY frontend/. .

RUN npm run build

EXPOSE 80
CMD ["npm", "run", "preview"]
