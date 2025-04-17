Docker packages individual applications and their dependencies into containers, ensuring they run consistently across different environments. Docker Compose takes this a step further by helping you manage multiple containers as a single unit, ensuring they work together seamlessly.

- Docker is a tool for running applications in an isolated environment.
- Similar to virtual machine without the copy of OS and with less memory.
- Allows app to run in exact same environment you want
- It helps it easy to package the application

# **Docker is a container**

A container are an abstracttion at the app layer that **packages code and dependencies** together. Multiple containers can run on the same machine and share the OS kernel with other containers, each running as isolated processes in user space.

# ** Docker Image**

Image is a template for creating an evironment of your choice. It is a snapshot and has everything needed to run your apps. It has your OS, software and app code.

Container is nothing but a running instance of an image. A container is created when an image is run. The container then runs the executable application. The images are in the docker hub which is a registry. https://hub.docker.com/ - A place where the images are stored for public access.

Open Docker Desktop before you do anything with docker- Make sure its running.

# ** Exposing Port **

Our computer acts a host from where we run the docker from. Suppose we have a container running (say nginx), then the container expose its port(say 80/tcp). When the host want to access the container that exposed it port the host issues a request to the container using url address localhost:8080. What is expected is - Port 8080 on host should maps to port 80 on the nginx container to access the application on the container.

To do so, we need use -p localhostport:containerport while running the container i.e., docker run -d -p 8080:80 image:tag

So from now, the localhost port 8080 maps to container port 80. Whenever we type localhost:8080, we are able to access our container.

Once mapped, to access that container, we simply go to the web browser and using this address localhost:8080 to access it.

Note that localhost port number need not to be 8080. It can be any number.

We can also expose multiple ports to the same container.

    docker run -d -p 8080:80 -p 8085:80 image:tag

# ** Volumes **

Volume is created on container which allows us to share the data (such as files and folders) between host and container, and also between containers.

Data created in host's file system (in docker area) , also presents in docker container(inside volume) and vice versa.

- bind mount between container and file system.
- tmpfs mount between container and memory

### **Commands**

- <docker>
- <docker --version>
- <docker images> # shows the locally downloaded images with tag, image id, repository name, size, created time
- <docker ps> # List running containers
- <docker ps --all/-a> # List all running & not running containers
- <docker ps --help>
- <docker ps --format="ID\t{{.ID}}\nNAME\t{{.Names}}\nIMAGE\t{{.Image}}\nPORTS\t{{.Ports}}\nCOMMAND\t{{.Command}}\nCREATED\t{{.CreatedAt}}\nSTATUS\t{{.Status}}\n"> # for pretty printing the docker ps

  - We can also set env variable to save this format and use it later
    - <set FORMAT="ID\t{{.ID}}\nNAME\t{{.Names}}\nIMAGE\t{{.Image}}\nPORTS\t{{.Ports}}\nCOMMAND\t{{.Command}}\nCREATED\t{{.CreatedAt}}\nSTATUS\t{{.Status}}\n"> -- For Windows
    - <export FORMAT="ID\t{{.ID}}\nNAME\t{{.Names}}\nIMAGE\t{{.Image}}\nPORTS\t{{.Ports}}\nCOMMAND\t{{.Command}}\nCREATED\t{{.CreatedAt}}\nSTATUS\t{{.Status}}\n"> -- For MAC
  - <docker ps --format=$FORMAT> -- for mac
  - <docker ps --format=%FORMAT%> -- for windows

- <docker container ls> # list running containers
- <docker images ls> # list images
- <docker image inspect imagenameorid> # we can find all the metadata realted to image. like exposed ports and eveything

  - <docker image inspect image_name_or_id | findstr "ExposedPorts"> # for cmd
  - <docker image inspect image_name_or_id | Select-String "ExposedPorts"> # for powershell
  - <docker image inspect image_name_or_id | grep -i "exposedports"> # for linux

- <docker run image:tag> # Create and run a new container from an image
- <docker run -d image:tag> # Create and run a new container from an image in detach mode (console cursor is available after running)
- <docker run -d -p 8080:80 image:tag> # run container while exposing the port, it can be port 8080 or any other number like 8085 and so on
- <docker run -d -p 8080:80 -p 8085:80 image:tag> # exposing multiple ports to same container
- <docker pull image> # Download an image from a registry
- <docker pull image:tag> # Download an image from a registry
- <docker rmi imageid> # Remove the image saved locally, use -f for forcefully delete
- <docker rmi image:tag> # Remove the image saved locally
- <docker stop containerid/containername> # To stop the running container
- <docker start containername/containerid> # To start the stopped container
- <docker rm containerid/containername> # To delete the container
- <docker rm -f $(docker ps -aq)> # To delete all the containers in one go (only works when containers are not running)
  - In windows
    - for cmd its: <for /F %i in ('docker ps -aq') do docker rm -f %i>
    - for powershell its: <docker ps -aq | ForEach-Object { docker rm -f $\_ }>
- <docker run --name urcontainername -d -p 8080:90 image:tag> # name your container as you like
- Creating a volume and bind mount a volume i.e mapping host files to container volume
  <docker run --name mycontainer -v hostsourcefilepath:destinationcontainervolumepath:access -d -p 8080:80 image:tag>
  - Suppose, we want to share index.html of our website application
  - Go the folder path in the host system that you want to share to the container
    - cd E:\Learning\ML\Learning Practice\Code\Computer Vision\pathology\docker\website - Here index.html resides
  - Start the container here while mounting the source folder of host system to the container
    - <docker run --name mywebsite -v "%cd%:/usr/share/nginx/html:ro" -d -p 8080:80 nginx:latest> # Note it is readonly access - ro
    - In windows cmd its %cd%, powershell its ${PWD}, in mac its $(pwd)
    - Make sure to have quotes for the paths when has spaces, otherwise not needed.
- <docker exec -it containername command> # executes the command in the running container in interactive mode

  - <docker exec -it mywebsite sh> # # runs the bash, useful for minimal containers like alpine based
  - <docker exec -it mywebsite /bin/sh>
  - <docker exec -it mywebsite bash> # runs the bash and we will be inside the container(linux env).
  - <docker exec -it mywebsite /bin/bash>
    Now we can check if our volume in the container /usr/share/nginx/html has the mounted index.html
  - When we run https://localhost:8080, the index.html is executed and displayed
    - Note: docker run does not specifically look for index.html unless the containerized application is configured to do so. It depends on the base image and the application running inside the container. For ex, if you're running an NGINX container (nginx:latest), it will, by default, serve index.html from /usr/share/nginx/html/. If you're using Apache HTTP Server (httpd), it also expects an index.html or another default file in /usr/local/apache2/htdocs/. For a custom application (e.g., Python Flask, Node.js, etc.), it will serve whatever entry point the application is configured to use.
  - exit # to exit the container
  - We cant create files in the container when we have ro access
  - <docker run --name mywebsite -v "%cd%:/usr/share/nginx/html" -d -p 8080:80 nginx:latest>
    - <docker exec -it mywebsite bash>
    - <cd /usr/share/nginx/html>
    - <touch myfile.html> # able to create now and also created automatically in host file system
  - Mount volumes from the specified container i.e we can share the same volume between 2 containers so that every file/folder in one container exists in another
    - <docker run --name newcontainer --volumes-from oldcontainer -d -p newlocalport:containerport image>
      - <docker run --name mywebsite-copy --volumes-from mywebsite -d -p 8081:80 nginx>

- ## **Dockerfile**

  - Dockerfile helps to build our own images instead of building the existing images like nginx.
  - Usual practice: Instead of mounting the host files on to container volume, we have to build the image that has all the files we have in the host sytsem and then we can just create a container from our image.
  - Image is a snapshot of your app. It should contain everything that your app needs (OS, libraries, software, app code)

    - Create Dockerfile inside the root directory of the project in VScode
    - <FROM existedknownimage:tag> # always start with "FROM existedknownimage". We always start with known image. we dont build from scratch
      - <FROM nginx:latest>
      - <FROM node:latest>
    - <WORKDIR /Folder/in/container>
      - Eg: <WORKDIR /app> # Set the work directory inside the container. Creates/Use a folder called app inside the container. Any commands that follow this ll have the app as their current working directory
    - <ADD . pathincontainer> # Add all the files in the current directory to the path of the container
      - Eg: <ADD . /usr/share/nginx/html> # # add everything in the current directory to the container directory /usr/share/nginx/html
      - Eg: <ADD . .> # when you set workdir..add all the contents from local to workdir in container
    - <RUN yourcommand> # allows us to execute the command in the container
      - Eg: <RUN npm install> # to install all dependencies from package.json
    - <CMD yourexectioncommand> # to run the command in cmd
      - Eg: <CMD node index.js>

- **Build Image and Run container from it**

  - Make sure to navigate to the project root directory in cmd
  - <docker build --tag imagename:tagname pathofyourdockerfile>
    - Eg: docker build --tag website:latest .
  - <docker run --name containernameasuwish -d -p localport:containerport builtimage:tag> # to run a container from our built image
    - Eg: <docker run --name website -d -p 8080:80 website:latest>
    - Eg: <docker run --name userjsapp -d -p 3000:3000 userjsapp:latest>

- **.dockerignore**
  - This file contains the files, folders paths that we dont want to have in the built image. For eg, in the Dockerfile, we have <RUN npm install>, so the node_modules ll e created again when we run the container, so we dont need our local node modules to be in our docker container. We also ignore Dockerfile in our image building
  - Eg .dockerignore file
    <!--
    node_modules
    Dockerfile
    .git
    *.err
    /folder/**
    logs
    -->
- **Caching and Layers**
  Docker optimizes image builds using layering and caching, which helps reduce build time and resource usage.

**Docker Layering**
Docker images are made up of multiple read-only layers stacked on top of each other. Each instruction in a Dockerfile creates a new layer. When you run a container, Docker adds a writable container layer on top of the image layers.

<!--
# Base layer
FROM python:3.9
# New layer: Install dependencies
RUN pip install flask
# New layer: Copy application files
COPY app.py /app/app.py
# New layer: Set working directory
WORKDIR /app
# Final layer: Define the command to run the app
CMD ["python", "app.py"]
 -->

Each of these instructions creates a separate layer. If a layer changes, Docker will rebuild that layer and all layers above it.

**Docker Build Cache**
Docker caches layers to speed up builds. When rebuilding an image, Docker reuses unchanged layers instead of recreating them.
**How Caching Works**
If Docker detects a matching existing layer, it uses the cached version. If a layer changes, Docker invalidates all subsequent layers and rebuilds them.

<!--
FROM ubuntu:latest
RUN apt-get update && apt-get install -y curl  # Layer 1
COPY myscript.sh /usr/local/bin/myscript.sh    # Layer 2
RUN chmod +x /usr/local/bin/myscript.sh        # Layer 3
 -->

If only myscript.sh changes, Docker reuses Layer 1 but rebuilds Layer 2 and Layer 3.
If the apt-get install command changes, Docker invalidates all layers after Layer 1, causing a full rebuild.

**Optimizing Caching for Faster Builds: Best Practices for Efficient Docker Caching**

- Order Dockerfile instructions from least to most frequently changing.
- Place <RUN apt-get install, pip install, or npm install> before copying application files.
- Use .dockerignore to exclude unnecessary files (e.g., logs, .git).
  - Helps prevent cache invalidation due to unrelated file changes.
- Use multi-stage builds to reduce the final image size.

  - Helps avoid storing unnecessary intermediate files.

- Now with these parctices, the building image wont take much time to render the updated changes in the application.

- Examples
  <!--
    FROM node:latest
    WORKDIR /app
    ADD . .
    RUN npm install
    CMD node index.js
  -->

  to

  <!--
    FROM node:latest
    WORKDIR /app
    ADD package*.json ./
    RUN npm install
    ADD . .
    CMD node index.js
  -->

  # Bad: Frequent file changes break caching

  <!--
  FROM python:3.9
  WORKDIR /app
  COPY . /app/
  RUN pip install -r /app/requirements.txt
  CMD ["python", "app.py"]
  -->

  # Good: Dependencies first (cached longer)

  <!--
  FROM python:3.9
  WORKDIR /app
  COPY requirements.txt /app/
  RUN pip install -r /app/requirements.txt
  COPY . /app/
  CMD ["python", "app.py"]
  -->

## **Reducing Image Sizes**

- Alpine tags are small in size. Check in docker hub for the base image alpine tags instead of pulling from latest tag - https://alpinelinux.org/ , https://hub.docker.com/
- <docker pull requiredimage:lts-alpine> or <docker pull requiredimage:alpine> # lts stands for latest, here we pull alpine version of required image
  - <docker pull node:lts-alpine>
  - <docker pull nginx:alpine>
- We can also switch our own built image to alpine version to reduce its size. Just change the <FROM knownimage:alpine> in the Dockerfile and rebuild and run.
  - Eg: <FROM node:alpine>
    <FROM nginx:alpine>

## ** Tags, Versioning and Tagging**

- We have the ability to pull or the required image version or even alpine version by giving the versions explicitly required in Dockerfile
  - Eg: <FROM image:version-alpineversion>
  - Eg: <FROM node:23.10.0-alpine3.20>
- **Tagging our own images**
  Tagging images as latest always when building the image with another versions from DockerFile leads to making previous versioned repositories & tags as none. This is cuz it overrides the existing container with the same image name and latest tag. Can be seen with <docker image ls>. Hence it good practice to tag differently when building using other version and this makes sure to not loose the previous versions. This essentially helps us to rollback to previous version easily when there are code breakings.
  - <docker tag image:oldertag image:newertag>
    - <docker build -t website:latest .> # initial latest code
    - <docker tag website:latest website:1> # creating the version from the initial latest version
    - <docker build -t website:latest .> # you had some changes and built the latest image again
    - <docker tag website:latest website:2> # Now you created another version with tag 2. It contains latest changes that we had from latest tagged image. Now image with tag 1 have previous version
    Now latest and 2 have a same versioned image - similar copy instead of getting overridden. We can just run <docker run...> with different ports for all the 3 version and see how 3 versions differ from UI
      - <docker run --name website-latest -d -p 8080:80 website:latest>
      - <docker run --name website-2 -d -p 8083:80 website:2>
      - <docker run --name website-1 -d -p 8082:80 website:1>

## ** Docker Registries - Pushing to and Pulling from the registry**

Registry is like a repository where the public and private images are stored. Instead of having our images in the host sytem, we can push it to repository. It can be either Docker hub or Amazon EC2 Registry or https://quay.io/

- If its docker hub:

  - First login to the registry using UI
  - Go to My Hub and click on Create a repository
  - Give repo name, access as public or private
  - First rename the local images that we want to push with username/reponame and desired tag. and then push. Make sure to name your repository appropriately
  - <docker tag local-image:tagname username/reponame:desiredtagname>
  - Execute <docker login> and give the credentials in cmd
  - <docker push username/reponame:desiredtagname>
    - Eg:
      - <docker tag website:latest bhavanamalla/mywebsite:latest>
      - <docker tag website:1 bhavanamalla/mywebsite:1>
      - <docker tag website:2 bhavanamalla/mywebsite:2>
      - <docker push bhavanamalla/website:latest>
      - <docker push bhavanamalla/website:1>
      - <docker push bhavanamalla/website:2>
  - <docker pull username/reponame> # by default pulls the image with latest tag, else can also give desired tag
    - Eg:
      - <docker pull bhavanamalla/website>
      - <docker pull bhavanamalla/website:1>

## ** Debugging Containers **

- <docker inspect containername/id> # to inspect the docker config in json format
- <docker logs containername/id> # to get the logs of the docker
- <docker logs -f containername/id> # to get the logs of the docker and -f or --follow allows us to make the docker log alive when running the app so that we can see logs in live interactive mode

### ** Docker Compose**

Docker Compose is a tool that allows you to define and manage multi-container Docker applications using a simple YAML file (docker-compose.yml). Instead of running multiple docker run commands manually for every app/service in your application, you can define all your services in one place and start them with a single command.

Why Use Docker Compose?

- Easier management – Define multiple services in one file <docker-compose.yml>
- Simplifies networking – Containers can communicate easily
- Volume & environment management – Define everything in YAML
- One-command startup – <docker-compose up> starts everything

Usually an app contains several services and for each service we need a separate container for it deploy or work in isolation. We also need these services to interact with each other. Like Login service, ui service, messaging service, multiple db services, middleware service, logging service, cache service and so on. To establish the containers to interact with each other, we need a network. When you dont have docker-compose.yml, we need to create network manually and we need to make the containers of different services of our application to interact with each other within that network which is cumbersome and tedious. We have start the containers parallely and also need to stop them one by one when a update is required.

Imagine you have an app where you have mongo db service and mango-express service(which is just an UI tool to interact with mongo db). For them to interact with each other, we need to follow the below steps. Here we also environment variables to give db login user and password with "-e" or "--environment"
**Creating Network and running containers within the network**

- <docker network create networkname>
  - Eg: <docker network create mongo-network>
- <docker network ls>
- <docker run --network networkname --name containername -e envvarible=something -e envvariable=something -d -p localhostport:containerport image:tag>

  - Eg: # https://hub.docker.com/_/mongo
  - <docker run --network mongo-network --name mongodb -e MONGO_INITDB_ROOT_USERNAME=admin -e MONGO_INITDB_ROOT_PASSWORD=pass -d -p 27017:27017 mongo>
  - <docker exec -it mongo mongosh -u admin -p pass --authenticationDatabase admin # command to check if username and pwd establish connection
  - <docker run --network mongo-network --name mongoexpress -e ME_CONFIG_MONGODB_ADMINUSERNAME=admin -e ME_CONFIG_MONGODB_ADMINPASSWORD=pass -e ME_CONFIG_MONGODB_SERVER=mongo -d -p 8081:8081 mongo-express> # note that we gave our mongodb username pwd and the container name that we gave for mongodb container in mongoexpress for mongoexpress to connect to mongodb

  - When we access https://localhost:8081, we access the mongo-express through which we login to mongo db.
  - We can create database collection, documents inside the db. But they are not persistent. When we remove the container using docker rm ... command, we are going to loose the state as docker is stateless. So, if you want created data to be persistent, instead of deleting, we can use <docker stop containername> and <docker start containername>
  - FYI, We can remove docker network if you want using <docker network rm networkname>

Note: Running containers with "--rm" flag is good for those containers that you use for very short while just to accomplish something, e.g., compile your application inside a container, or just testing something that it works, and then you are know it's a short lived container and you tell your Docker daemon that once it's done running, erase everything related to it and save the disk space.

**Structure of dockercompose.yaml**

<!--
version: "3.1" # docker compose version that is compatible with your installed docker
services: # you define your containers for all your app service here
  containername:
    image: imagename:tag
    ports:
      - host:container
    environment:
      envvar: val
      envvar: val

  containername:
    image: imagename:tag
    ports:
      - host:container
    environment:
      envvar: val
      envvar: val
      envvar: val
    depends_on:
      - "anothercontainername" # when we know this service depends on another service,, eg: mongo-express needs mongodb to start first. "<- vals>" means its list of vals
-->

<!-- Eg: mongo-services.yaml

version: "2.32.4"  # this attribute not needed
services:
  mongodb:
    image: mongo
    ports:
      - 27017:27017
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: pass

  mongoexpress:
    image: mongo-express
    ports:
      - 8081:8081
    environment:
      ME_CONFIG_MONGODB_ADMINUSERNAME: admin
      ME_CONFIG_MONGODB_ADMINPASSWORD: pass
      ME_CONFIG_MONGODB_SERVER: mongodb
    depends_on:
      - "mongodb"

 -->

- <docker-compose -f file.yaml up -d> # run the container in detach mode
- <docker-compose -f file.yaml down> # Dont use it when you want the state to be persistent. This will stop the containers and also delete them. Instead use stop, which allows them to stop running containers but dont delete the container once and for all.
- <docker-compose -f file.yaml stop>
- <docker-compose -f file.yaml start>
- Note: When we up the docker-compose, the container and network names are created with the format rootdir-containername-version, and network as rootdir_default Eg: website-mongodb-1, website-mongoexpress-1, website_default
- We can also rename the project name to your desired name instead of rootdir as we like using -p desriedname or --project-name desriredname
  - <docker-compose --project_name desirename -f file.yaml up -d>

### ** Dockerfile and docker-compose yaml file**

- To build our own image for our application, we create Dockerfile and <docker build -t image:tag pathtoDockerfile>. We can also do this directly in yaml file to build image for our app from Dockerfile
- In our yaml file, we need to add service for our application service now for us to build image and run container.
  <!--
  services:
    containername:  # container name of our app service that we want to containerize, say for server.js
      build: .  # specify the path where our DockerFile is located
      #rest are almost similar
      ports:
        - hostportwhereappstarts:containerport
      environment:
        envvarsetinapp: val
        envvarsetinapp: val
      depends_on:
        - somecontainer  # if required, When we dont need the app to connect to any db or any other service while loading its ui, then it may not required
   -->

  - Eg:
    <!-- services:
          myapp:
            build: .
            ports:
              - 3000:3000
            environment:
              MONGO_DB_USERNAME: admin
              MONGO_DB_PWD: pass
      -->
  - Eg: Dockerfile
    <!--
     FROM node:20-alpine
     RUN mkdir -p /home/app
     WORKDIR /home/app
     COPY package*.json ./
     RUN npm install
     COPY . .
     EXPOSE 3000  # imp
     CMD [ "node", "server.js" ]
     -->

    Eg: mongo-service.yaml
     <!-- 
      services:
        myapp:
          build: .
          ports:
            - 3000:3000
          environment:
            MONGO_DB_USERNAME: admin
            MONGO_DB_PWD: pass
    
        mongodb:
          image: mongo
          ports:
            - 27017:27017
          environment:
            MONGO_INITDB_ROOT_USERNAME: admin
            MONGO_INITDB_ROOT_PASSWORD: pass
    
        mongoexpress:
          image: mongo-express
          ports:
            - 8081:8081
          environment:
            ME_CONFIG_MONGODB_ADMINUSERNAME: admin
            ME_CONFIG_MONGODB_ADMINPASSWORD: pass
            ME_CONFIG_MONGODB_SERVER: mongodb
          depends_on:
            - "mongodb"
    
      -->

- Note: if the app is node based js app, Make sure to run <npm init -y> in the local working directory to create package\*.json files in the local working directory before running docker compose.
- Then we use yaml file and <docker-compose -f file.yaml up -d> to run the container for our application
- Note: When facing module not found issues while trying to up the container, execute the up command with build option to rebuilt the image and not to take image from cache. May be --no-cache also works
  <docker-compose -f mongo-services.yaml up --build -d>
