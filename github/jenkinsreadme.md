### Jenkins

Jenkins is an open-source automation server used for continuous integration and continuous deployment (CI/CD). It helps automate software development processes like building, testing, and deploying applications. It supports plugins to extend functionality (e.g., Git, Docker, Kubernetes).

**Key Concepts in Jenkins:**

- Jobs & Pipelines – Define tasks that Jenkins will execute.
- Build Triggers – How Jenkins starts a job (push, PR, schedule).
- Nodes & Agents – Run jobs on distributed machines.
- Plugins – Extend Jenkins capabilities (e.g., Slack notifications, GitHub integration).

**Install Jenkins using Docker**

- <docker run -d --name jenkins -p 8080:8080 -p 50000:50000 --restart=on-failure jenkins/jenkins:lts> # pull the jenkins long term support image and create contaier
- Go to <https://localhost:8080> to access Jenkins and follow the setup wizard. Give the Admin password that Jenkins display when pulling the image. If you forgot the copy the initial admin Password, then go to the running docker container bash in interactive mode <docker exec -it contianername bash> and go to <cd /var/jenkins_home/secrets> and <cat initialAdminPassword>. You can exit the container using exit command.
- You can give username and pwd for you to login with your creds later on instead of admin default pwd
- Install initial suggested plugins (Git Plugin (for integrating GitHub, GitLab, etc.), Pipeline Plugin (for writing CI/CD workflows), Docker Plugin (for containerized deployments))

**Create Your First Jenkins Job**

- Go to Jenkins Dashboard → New Item
- Select Freestyle Project
- Add Git Repository URL (e.g., GitHub). Say we have some python file with print("Hello World")
- When the git repo url failed to authenticate, then we need to add Pat of github to Jenkins Credential Manager before creating the git based job.
  https://medium.com/@maheshwar.ramkrushna/integrating-jenkins-with-gitlab-using-personal-access-tokens-for-secure-ci-cd-automation-1b9d73f517c8
- Go to the job -> Configure and give the git url after setting pat.
- Add a build step: (give the command of your project entry point)
  If you are running from docker, then its linux env. Give <python yourfile.py> as Execute shell command, if the jenkins in windows, then execute as Execute Windows batch command. Also, Make sure that Python is installed in the Jenkins environment, and the Python executable is available in the system path. You can check in your docker linux container <python --version>. If not installed, <apt-get update && apt-get install -y python3 python3-pip> in container.
  - We can also build our own image with python installed in Jenkins using Dockerfile
    <!--
    # Use the official Jenkins image as a base image
    FROM jenkins/jenkins:lts
    # Switch to root user to install packages
    USER root
    # Update the package list and install Python
    RUN apt-get update && apt-get install -y python3 python3-pip
    # Switch back to Jenkins user
    USER jenkins
    # Continue with the rest of your Dockerfile...
    -->
    But the best way is to the best way to do it is to write a shell script for what you need to do and then call this shell script from Jenkins as a "shell command" step. It's simple, it puts you in control and it gives you everything you need. You are not limited by what Jenkins provides, it works great with virtualenv and your developers can run the same script on their computers which is also extremely helpful.

You can commit this script as part of the project in your repository.

- Poll SCM
  - Give the cron job command here Eg: <\* \* \* \* \*>. See cron expression to know <MINUTE HOUR DOM MONTH DOW>
- Click Build Now and check console output by clicking on the build url under builds.
- Note: For a Git push trigger in Jenkins, you don't need a cron expression. Instead, use webhooks from your Git hosting service (like GitHub) to notify Jenkins to trigger the build. The cron expression is for time-based scheduling, like running jobs at specific intervals.

Note: Jenkins use Groovy script to run pipelines
