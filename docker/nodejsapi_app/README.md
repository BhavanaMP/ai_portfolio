- Make sure you download node js from website
  - Helps to develop javescript based applications
- Make sure the npm is installed
  - **npm install** By default, it will install all node modules listed as dependencies in **package.json**
- Express api is used to api's seamlessly - Creates an Express application. The express() function is a top-level function exported by the express module.

  - https://expressjs.com/en/starter/installing.html
    - Go to the current project working directory in cmd and execute **npm init**
      This command prompts you for a number of things, such as the name and version of your application. For now, you can simply hit RETURN to accept the defaults for most of them, with the following exception: entry point: (index.js). Enter app.js, or whatever you want the name of the main file to be. If you want it to be index.js, hit RETURN to accept the suggested default file name.
    - Now, install Express in the current directory and save it in the dependencies list.
      **npm install --save express**

- Hello World
  Below is essentially the simplest Express app you can create. It is a single file app — not what you’d get if you use the Express generator, which creates the scaffolding for a full app with numerous JavaScript files, Jade templates, and sub-directories for various purposes.
     <!--
         const express = require('express')
         const app = express()
         const port = 3000
         app.get('/', (req, res) => {
         res.send('Hello World!')
         })
         app.listen(port, () => {
         console.log(`Example app listening on port ${port}`)
         })
     -->
  We will create this code in app.js or index.js whatever you like inside our directory where npm in initiated. This app starts a server and listens on port 3000 for connections. The app responds with “Hello World!” for requests to the root URL (/) or route. For every other path, it will respond with a 404 Not Found.
- Run the app with the following command: **node app.js**
- Then, load **http://localhost:3000/** in a browser to see the output.

- Use the application generator tool, **express-generator**, to quickly create an application skeleton. https://expressjs.com/en/starter/generator.html
  - **npx express-generator**
  - **express --help**
  - **express --view=pug myapp**
    For example, the above command creates an Express app named myapp. The app will be created in a folder named myapp in the current working directory and the view engine will be set to Pug
  - **cd myapp**
  - **npm install** # Then install dependencies
  - **DEBUG=myapp:\* npm start** # ON Mac
  - **set DEBUG=myapp:\* & npm start** # On Windows Command Prompt
  - Then, load **http://localhost:3000/** in your browser to access the app.
  - The generated app has the following directory structure: 7 directories, 9 files
    ├── app.js
    ├── bin
    │ └── www
    ├── package.json
    ├── public
    │ ├── images
    │ ├── javascripts
    │ └── stylesheets
    │ └── style.css
    ├── routes
    │ ├── index.js
    │ └── users.js
    └── views
    ├── error.pug
    ├── index.pug
    └── layout.pug
