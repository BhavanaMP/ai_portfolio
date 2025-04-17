const express = require("express");
const app = express();
const port = 3000;

// app.get("/", (req, res) => {
//   res.send("Hello World!");
// });

app.get("/", (req, res) => {
  res.json([
    {
      name: "First Json App",
      gmail: "some@xyz.com",
    },
    {
      name: "Second Json App",
      gmail: "some2@xyz.com",
    },
    {
      name: "Third Json App",
      gmail: "some3@xyz.com",
    },
  ]);
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
