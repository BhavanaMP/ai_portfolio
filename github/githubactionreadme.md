### **GitHub Workflow and Actions**

Like Jenkins, Github Actions is a popular CI/CD pipeline that is integrated directly in Github repo.

A Workflow in Github is a a configuable automated process and is a collection of correlated Github Actions defined in a single file to control sequence of events or tasks. A workflow is a single file; you may have multiple workflows for different things. An Action is a reusable packaging of a workflow. You create an action when you want to run the same workflow in multiple repositories. A workflow can run one or more jobs. Workflow triggers are events that cause a workflow to run (given for 'on' attribute).

In a workflow file, you define two things:

- What conditions trigger this? For instance, a merge onto master.
- What things should happen?

The "things that happen" are technically called jobs, and are then composed of steps.

Workflow is like a playlist where action is a song in that playlist.

Github workflows are written and saved as yaml files and saved under <.github/workflows/desiredname.yaml> in the repo

## Basics

- To create a workflow, Go to repo -> Actions -> Set up a workflow yourself or use search option

## Github Actions Workflow components

**Workflows:**
Automated processes define in your repository that coordinate one or more jobs, triggered by events or on a schedule. It’s composed of jobs and runs in response to specific events (like push, pull request, schedule, issues, pull_requests). Defined in .github/workflows directory as a YAML file.

**Actions:**
Reusable tasks(scripts or commands) that perform specific jobs within a workflow. Basically they perform tasks, like setting up an environment, building code, running tests, or deploying applications.Can be created by GitHub or the community and shared via the GitHub Marketplace.

**Jobs:**
A set of steps that execute on the same runner. Jobs can run sequentially or in parallel depending on the workflow’s configuration. Each job runs in its own virtual environment.

**Steps:**
Individual tasks within a job. Each step can run a script, an action, or a shell command. Steps are executed in order within a job.

**Runs:**
The execution of a workflow on GitHub’s infrastructure, triggered by events like push, pull request, or manual triggers.

**Runners:**
Virtual machines (or self-hosted machines) where workflows and jobs are executed. GitHub provides hosted runners for different operating systems (Ubuntu, Windows, macOS), or users can set up their own self-hosted runners.

**Marketplace:**
A place to find and share reusable actions created by the GitHub community or official sources. Helps you avoid reinventing the wheel by using pre-built actions for common tasks.

## Simple workflow

- <name: Workflow name> name attribute is saved as workflow name
- <on: [list of event triggers]>. on attribute specifies the what event will trigger this workflow. Eg: <on: [push]>. So when push event happens to the repo, it will trigger this workflow. Github has 35+ event triggers such as push, pull, merge, pull_request_target, issues, releases, schedule

<!--greeting.yml
name: Simple workflow # this show as workflow name
on: [pull_request_target, issues]

jobs:
    greeting: # job name
        runs-on: ubuntu-latest
    permissions:
        issues: write
        pull_requests: write
    steps:
    - uses: actions/first-interaction@v1  # This the repo link of existing action. we can use existing actions or we can create our own actions
      with: # with is the kw used to input the params needed by the action
        repo-token: ${{ secrets.GITHUB_TOKEN }}
        issue-message: "Message that will be displayed on users' first issue"
        pr-message: "Message that will be displayed on users' first pull request"
 -->

<!--scheduleworkflow.yml # schedule can use a 5 asterisk cron expression to trigger a workflow at a specific time or day
name: Schedule workflow
on:
  schedule:
    - cron: '30 5 * * 1,3'
    - cron: '30 5 * * 2,4'

jobs:
    test_schedule:
        runs-on: ubuntu-latest
    steps:
        - name: Not on Monday or Wednesday
          if github.event.schedule != '30 5 * * 1,3'
          run: echo "Skip this step on Monday and Wednesday"
        - name: Everytime
          run: echo "This step will always run"
 -->

 <!--multieventworkflow.yml #make sure the yaml while is there in all the branches that you wanted to work on in this workflow
name: multievent workflow
on:
    push:  # default is master when not specified branches
        branches:
            - master
            - dev
    pull_request:  # default is master when not specified branches
        branches:
            - master

jobs:
    test_multievent:
        runs-on: ubuntu-latest
        steps:
            - name: "Echo Basic information"
              run: |
                echo "REF: $GITHUB_REF"
                echo "JOB ID: $GITHUB_JOB"
                echo "Action: $GITHUB_ACTION"
                echo "Actor: $GITHUB_ACTOR"
 -->

- We can disable workflow if we dont need it from Actions UI, select the action and click on dropdown top left to disable workflow.
- We can also disable all workflows by going into github dev mode. Go to code window of the repo and click period in your keyboard to get into dev mode. Eg: https://github.dev/username/reponame. Move or delete those yaml files under workflows, and this deletes the workflows from actions UI.
- Thw workflow runs history also can be deleted from UI itself directly of each workflow
