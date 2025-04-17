simple java script application that connects to mongodb and get data dynamically from DB and displays in the browser. The Javascript application is containerized

- In this, the goal is to containerize the app using DockerFile and docker-compose. Dockerfile helps to build our own images instead of building the existing images. And docker-compose up helps to run the container.

- Make sure to run <npm init -y> in the local working directory to create package\*.json files in the local working directory before running docker compose.
- Make sure to install express and mongodb using <npm install express> <npm install mongodb> to add it to package.json

- When facing module not found issues, run
  <docker-compose -f mongo-services.yaml up --build -d>
