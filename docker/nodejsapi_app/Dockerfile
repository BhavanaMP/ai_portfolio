FROM node:23.10.0-alpine3.20
WORKDIR /app
ADD package*.json ./
RUN npm install
ADD . . 
CMD node index.js

# in cmd in current projects local workingdir
# - docker build -t imagename:tag pathtodockerfile
# - docker run --name containernameasyouwish -d -p 3000:3000 imagebuilt:tag #note localhostport is 3000 as mapped in the index.js and expressjs listens on by default