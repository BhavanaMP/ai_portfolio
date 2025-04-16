### What is version control?

Version control is a system that **records changes to a file or a set of files over time** so that you can recall specific version later. These versions are **recorded in a repository** and can be recalled from the same. There are **Local, Centralized and Distributed** version control systems.(distributed workflow -> local repo's -> remote repo -> server where code is deployed)

### Why version control is needed?

- Helps the deveopers to easily collaborate in shared workspaces and access real time updates
- History of code modifications is saved
- Allows to rollback, reverse faulty update, reduce downtime and saves time

### What is a repository?

Storage space where the projects live. Can be a folder on local computer or can be storage space on another online host such as Github. **Github is a web-based git repository hosting service**

### What is Git?

**Git is a Distributed Version Control tool**. It supports distributed non-linear parallel workflows, allows the team of developers work together on same project even remotely and enables them to easily merge their changes into one source.

### Workflow of git

    Developers
    ↓
    Clone → Local Repo
                ↓
                Checkout
                ↓
    Working Local Copy
    ↓
    (Modify Files)
    ↓
    Staging Area ← Stage Changes
    ↓
    Commit Changes to Local Repo
    ↓
    Fetch (Check for Remote Changes)
    ↓
    Pull (Get Remote Changes amd merge) → Rebase (if needed)
    ↓
    Push
    ↓
    Remote Repo

- Clone – Developers clone the remote repository to create a local repository on their machine.
- Local Repository – The cloned repository serves as a working local copy of the project.
- Checkout to get the working directory
- Working Directory – Developers make changes to files in their local working copy.
- Staging Area – Changes are added to the staging area using git add, preparing them for commit.
- Commit – Staged changes are committed to the local repository using git commit.
- Fetch & Pull:
  - Fetch – Retrieves changes from the remote repository without merging them.
  - Pull – Fetches and automatically merges the latest changes from the remote repository.
- Rebase (if needed) – Developers can use git rebase to apply their local commits on top of the latest changes from the remote repository, ensuring a clean commit history.
- Push – Commits are pushed from the local repository to the remote repository using git push

![workflow](images/workflow.png)

### Requirements

- Download Git bash

### Commands

- <git init>
    Initiates a new empty local git repository. Here, .git will be created.
- <git config --global user.name "yourname">
- <git config --global user.email "yourmailid">
  Configs can be done once. Not needed everytime
- <git config --list>
- <git clone repourl>
    Creates a copy of remote repository on your local machine
- <git remote add origin repolink>
    Eg: <git remote add origin https://github.com/BhavanaMP/gitlearn.git>
    Lets you add or link a remote repository to your created local repository. It creates alias of your remote repo, named as origin, in your local repo.
- <git remote -v>
  Allows us check if the remote repo is linked to our local repo
- <git remote rename origin desiredname>
  Lets us rename the alias origin to your desired name
- <git remote remove branchname> or <git remote rm branchname>
  Removes the remote repository from our local repo
- <git branch>
  Shows which branch you are on currently and all local branches that exist.
- <git branch -a>
  Shows all local and repo branches
- <git branch branchname>
    Eg: <git branch local-dev>
    Creates a new branch in our local repository. It creates a branch from the source repo which we are currently on. So all commits in that source branch is copied to this new branch.
- <git branch -m newbranchname>
  Helps to rename the current local repo branch names.
- <git branch -m oldbranchname newbranchname>
  Helps to rename the given local repo branch names to new given name.
- <git branch -m oldbranchname newbranchname>
  <git push alias :oldbranchname newbranchname>
  Helps to rename the branch name in local repo and pushing it to remote repo renames the branch in the remote repo as well. Note that oldbranchname should exists in remote repo for it rename.
  Eg:
  <git branch -m test staging>
  <git push origin :test staging>
- <git branch -d branchname>
  Delete branch from the local repository
- <git branch -D branchname>
  Delete branch from the local repository forcefully
- <git push -d alias branchname> or <git push alias :branchname>
  Deleted the given branch from the remote repo. Eg: <git push -d origin bug4563> <git push origin :bug4563>
- <git checkout branchname>
    Eg: <git checkout  dev>
    Switch to that given existing branch
- <git checkout -b branchname>
  Eg: <git checkout -b dev>
  Allows us to create a new branch with given branch name switch to that new branch
- <git add "filename"> or <git add .>
  To add specified or all files to staging area before comitting
- <git rm --cached "filename">
  To unstage files from staging area
- <git commit -m "message">
  Commits your changes from staging area to your local repository
- <git commit --amend>
  git commit --amend command is a convenient way to modify the most recent commit. It lets you combine staged changes with the previous commit instead of creating an entirely new commit. It can also be used to simply edit the previous commit message without changing its snapshot. But, amending does not just alter the most recent commit, it replaces it entirely, meaning the amended commit will be a new entity with its own ref.
  Eg: <git commit --amend -m "anupdatedcommitmessage">
  Eg:
  # Edit hello.py and main.py
    <git add hello.py>
    <git commit> 
    # Realize you forgot to add the changes from main.py 
    <git add main.py>
    <git commit --amend --no-edit>
- <git stash> or <git stash create yourmessage>
  When we want to save the current changes in the working branch but want to go back to clean working branch. (in case of checkout to another branches)
- <git stash pop>
  Get back the stashed changes and also removes keep the copy in the stash entries list
- <git stash apply>
  Get back the stashed changes but keep the copy in the stash entries list
- <git stash list>
  List the modifications stashes away by this command
- <git show stash>
    To inspect or check the details of particular stash
- <git stash clear>
    Remove all the stash entries
- <git stash drop -q stashidnfromstash@{n}>
  Removes a single specified stags from the list of stash entries
- <git rebase branchnamefromwhichchnagestobereflected>
    Rebase helps in linear clean commit history. Rebase is used when changes made in one existing branch needs to be reflected in another branch. When we created a local branch from master at C2 commit, then we add a new commit C4 to this local branch, meanwhile someone did commit C3 to Master, then we can get those changes reflected in our local branch using rebase, such that it looks like the commits are made sequentially. C1, C2, C3, C4. We essentially changes the base from C2 commit to C3 commit of our branch. Eg: <git rebase master> we essentially rebasing our local branch base from master. Make sure to be on or checkout to the branch of which you want re-base its base. Golden rule is "Never rebase a branch that has already been pushed to a shared remote repository if others are working on it. Rebasing rewrites commit history, which can cause serious issues for collaborators." 
    **When NOT to Use Rebase?**
    - On public/shared branches. Why? It rewrites history, causing conflicts for others when they pull.
    - After pushing commits to a remote branch that others are using. Why? Force-pushing (git push --force) after a rebase overwrites history.
    - When you need to preserve exact commit history. Why? Rebase rewrites commits instead of merging them.
    - On a remote tracking branch (origin/main, origin/dev). Why? Remote branches should not be rebased; use merge instead.
    Safe Alternative: git merge Instead of rebase. If you're working in a team and don't want to rewrite history: 
    <git checkout featurebranch>
    <git merge master>  # Safe alternative to rebase - merges master into feature branch to make sure featurebranch is up to date with master branch

- <git merge branchnametobemerged>
    Lets us to merge changes from a branch to another branch. Checkout the branch to which you want to merge changes to. Now run the command with branchnametobemerged
    Eg: Our branchnametobemerged is dev. We want to merge this branch changes to test or master. Then we checkout to test or master and run <git merge dev>. This will also copy the commits of the merged branch as it is. Can be checked with git log
- <git push aliasofremoterepo branchname>
    Eg: <git push origin master>, <git push origin dev>
    Lets you push your local changes from your local repo to the remote repo branch. Note that origin is  the alias of remote repo.  Give credentials for remote repo if asked while pushing.
- <git push -f alias branchname>
  Eg: <git push -f origin dev>
  Lets you push your local changes from your local repo branch to the remote repo branch forcefully overwwriting other commits.
- <git push alias --delete branchname>
  Deletes the branch in the remote repo Eg: <git push origin --delete somebranch>
- <git pull alias branch>
    Lets you copy and merge all the files, commits from the branch of remote repo to your local repo working branch. Downloads and merges the changes into your current branch (equivalent to git fetch + git merge). Eg: <git pull origin master>. By default, <git pull origin> only updates the currently checked-out branch (HEAD). It does not update all branches.If you are on master, it pulls updates only for master. If you switch to another branch (dev, feature-branch), it will update only that branch when you run <git pull origin> again.
- <git fetch alias branch>
    Checks for new commits in the remote repository and Updates your local repository without merging any changes into your working branch unlike git pull which merges changes into working branch automatically. It downloads new commits, branches, and tags from the remote repository but does not modify your working directory. Basically Updates your local reference (origin/master, origin/dev, etc.) Eg: <git fetch origin>, <git fetch origin master>. You can now check new updates using <git log origin/main --oneline>. If you’re ready to update your branch after fetching <git merge origin/main> Merges fetched changes into the current branch or <git rebase origin/main> Rebases your branch on top of the fetched updates
  - <git fetch>	Fetches all updates from the default remote (origin).
  - <git fetch origin>	Fetches all branches and commits from origin.
  - <git fetch origin master>	Fetches only the master branch from origin.
  - <git fetch --all>	Fetches all branches from all remotes.
  - <git fetch --prune>	Removes stale remote-tracking branches that were deleted and no longer exist in the remote repo.
  - <git fetch --dry-run>	Shows what would be fetched without making changes.
- <git cherry-pick commithash>
    Selects specific commits from one branch and applies them to another. It is useful when you need to pick and apply individual commits instead of merging an entire branch.
    -<git cherry-pick commit>	Applies a specific commit to the current branch.
    - <git cherry-pick -n commit>	Picks the commit without committing (for manual changes).
    - <git cherry-pick -x commit>	Adds a reference to the original commit in the new commit message.
    - <git cherry-pick A B C>	Applies multiple commits A, B, and C.
    - <git cherry-pick start-commit^..end-commit>	Picks a range of commits (from start-commit to end-commit).
    - <git cherry-pick --abort>	Cancels an ongoing cherry-pick operation.
    - <git cherry-pick --continue>	Continues a cherry-pick after resolving conflicts.
- <git fork>
    A fork is a copy of a repository that you create under your own GitHub (or GitLab/Bitbucket) account. It allows you to experiment, modify, or contribute without affecting the original repository. Fork creates a copy on GitHub while clone downloads it locally. The forked repo has to be cloned to start working on it. Fork Purpose is to contribute to others projects while clone allows to work on a local copy of a repo.Fork has	no direct connection to original repo, but updates can be pulled. Clone is	directly connected to repo. When used fork, we need to manually pull changes from the upstream repo while for clone updates automatically when you <git pull alias>. To work on forked repo, clone the fork to your local machine so that you have local copy of fork.
    - <git clone https://github.com/YOUR-USERNAME/FORKED-REPO.git>
    - <cd FORKED-REPO>
    To keep your fork updated with the original repo, add the Original Repository as upstream. This sets the original repo as upstream.
    - <git remote add upstream https://github.com/ORIGINAL-OWNER/ORIGINAL-REPO.git>
    - <git remote -v>
    To fetch updates from the original repo inorder to sync your Fork with the Original Repository. This updates your fork with the latest changes.
    - <git fetch upstream>
    - <git merge upstream/main> or <git rebase upstream/main> # Or "master" if that's the branch name.
    After making changes, Push Changes to Your Fork
    - <git add .>
    - <git commit -m "Updated feature XYZ">
    - <git push origin main>
    To Contribute Back create a Pull Request. Go to GitHub -> Open your fork.-> Click "New Pull Request" -> Compare with the original repo with fork repo. basefork should be the upstream original repo  and base branch is upstream master. Head fork is your forked repo and compare branch is the master of your forked repo. After selection of repo and branches, submit the PR.The project maintainer will review and merge.
- <git restore --staged file_name>
  To move a file from the Git staging area back to the unstaged changes, use the command . This command removes the specified file from the staging area, leaving it as an unstaged modification. Alternatively, to unstage all files, you can use
  <git reset>
- <git reset --soft HEAD~1> To revert the last unpushed commit back to staging area
- <git diff localbranch remotebranch>
    Compare the remote branch origin/main with your local branch main. Eg: <git diff main origin/main>
    - <git diff --cached file1> # can also compare file with its remote counterpart
    <git diff --staged file1> # can also compare local file with its staged counterpart
    <git diff commitid1 commitd2> # can also compare 2 commits
    <git diff commitid1 file.txt> # can also compare commits with workspace file
    <diff localfile1 localfile2> # tests 2 different files in local repo
- <git remote show origin>
    shows all remote branches and their upstream tracking.
- <git remote prune origin>
    This will clean up the stale refs/remotes/origin/MASTER entry from remote tracking
- If you rename your default branch (say MASTER to master) from the UI, then to reflect changes in your local repo,
  <git branch -m MASTER master>
  <git fetch origin>
  <git branch -u origin/master master> # branch 'master' in local set up to track 'origin/master'
  <git remote set-head origin -a> # origin/HEAD set to master
- To update all branches
  <git fetch --all>
  <git checkout branch-name> # Switch to each branch
  <git pull origin branch-name> or <git rebase origin/branch-name> # Pull updates for that branch or Rebase instead of merging
  or we can use one liner for all the above:
  <git branch -r | grep -v HEAD | awk '{print $1}' | xargs -n 1 git pull origin>'
- If You Want to Reset & Pull Again
  <git checkout master>
  <git fetch origin>
  <git rebase origin/master>
  If things got messy and you want a fresh sync:
  <git reset --hard origin/master>
  <git pull origin master>
  Warning: This removes local changes.

- <git status>
  Gives the files tracking, history of modifications and stuff
- <git log>
  Gives the commit history along with commit hash id, author, commit message and time of our current branch
- <git log branchname>
  Gives the commit history along with commit hash id, author, commit message and time of our specified branch
- <git log --oneline>
  Shows commits logs in single lines
- <git log --oneline --graph --decorate --all>
- <git log branchname --oneline>
  Shows commits logs of a branch in single lines
- <git show commithashid>
  commithashid with first 7 to 8 letters of full commit hash id works. Press q to end.
- <git ls-files>
  Shows the files that are tracked by our local git repo. If new file is created and is not yet added to the staging using git add command, then git can't track that new file. Adding it to staging area helps git to start tracking this new files thereafter.
- <git archive branchname --format=zip -output=../filename.zip>
  Archives any branch from your local repository and save it as zip file that can be shared
- <git bundle create ../repo.bundl>
  Git archive sends only the specified file contents to the standard output. Git bundle creates a single file containing both file contents and history

![workflow](images/git_basic_commands.png)

**SSH or HTTPS:**

You don’t necessarily need SSH to push to GitHub. You have two options for authentication when pushing to GitHub:

- HTTPS
- SSH

**HTTPS (No SSH Required)**
If you cloned the repository using an HTTPS URL (e.g., https://github.com/yourusername/repo.git), you don’t need SSH. However, GitHub requires authentication via a Personal Access Token (PAT) instead of your GitHub password.

### Steps for HTTPS Authentication:

- Generate a GitHub Personal Access Token (PAT):
  Go to GitHub → Settings → Developer Settings → Personal Access Tokens
- Click "Generate new token (classic)" and select scopes like repo.
- Copy and save the token somewhere secure (it won't be shown again).
- Use the token instead of your password when pushing:
  <git push origin main>
- Git will prompt for username → Enter your GitHub username.
- Git will prompt for password → Paste the Personal Access Token.
- We can also Store the credentials to avoid re-entering the token. This saves credentials in Windows Credential Manager.
  - <git config --global credential.helper wincred>

2. SSH
   If you prefer SSH authentication, you must generate and add an SSH key to GitHub first.

### Steps to Set Up SSH on Windows:###

- Open Git Bash (not CMD or PowerShell).
- Generate an SSH key:
  <ssh-keygen -t rsa -b 4096 -C "your-email@example.com">
- Press Enter to save it in the default location (C:\Users\YourUsername\.ssh\id_rsa).
- Enter a passphrase (optional but recommended).
- Start the SSH agent and add the key:
  <eval $(ssh-agent -s)>
  <ssh-add ~/.ssh/id_rsa>
- Copy your SSH public key to the clipboard and add it to GitHub:
  <clip < ~/.ssh/id_rsa.pub>
- Go to GitHub → Settings → SSH and GPG keys → New SSH Key.
- Paste the key and save.
- Test your SSH connection to GitHub:
  <ssh -T git@github.com>
- If successful, you’ll see: Hi <your-username>! You've successfully authenticated, but GitHub does not provide shell access.
- Change Git remote to SSH (if you cloned using HTTPS):
  -<git remote set-url origin git@github.com:yourusername/repo.git>
- Push normally using SSH:
  <git push origin main>

### **Managing Git Credentials in Windows**

From Aug 2021, Git uses Personal Access Token to authenticate. So, we need to remove the existing creads that were saved before 2021 and add the PAT generated to windows credential manager for passwordless authentication when using https.

1. Open **Credential Manager** → **Windows Credentials** → Find `git:https://github.com`.
2. Click **Remove** to delete it if using the git hub login password

**Store a Personal Access Token (PAT) Instead of a Password:**

1. Generate a PAT:
   - Go to **[GitHub → Developer Settings → Personal Access Tokens](https://github.com/settings/tokens)**.
   - Generate a **fine-grained token** with required permissions.
2. Use PAT when Git prompts for a password when doing any git operations like clone,pull or push:
   <git clone https://github.com/your-repo.git>
   Username: your GitHub username
   Password: your PAT
3. Save credentials permanently in windows credential manager:
   <git config --global credential.helper manager>
4. Verify Git Credentials Configuration:
   <git config --list --show-origin> and Check if `credential.helper=manager` is set.

### **What to Do If Your GitHub Personal Access Token (PAT) Expires**

1. Generate a New Token from Github
2. Update Token in Windows Credential Manager
3. Clear Old Credentials (If Needed). If Git still uses the old token, clear it: <git credential reject https://github.com> Or manually remove the credential from Credential Manager and re-enter when prompted.
4. Update Token in Git Config (If Using a Saved Token in Config). If you stored the token in `~/.git-credentials`, update it:  
   <git credential reject https://github.com>
   <echo "https://<your-username>:<new-token>@github.com" > ~/.git-credentials>
5. Test Authentication
   <git pull origin branch>. If it works, you're good to go!

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
- Add a build step: (give the command of your project entry point)
  <python yourfile.py>
- Poll SCM
  - Give the cron job command here Eg: <\* \* \* \* \*> This 5 arterisk means run the build when there is a change
- Click Build Now and check console output.

Note: Jenkins use Groovy script to run pipelines

### **GitHub Workflow and Actions**

A Workflow in Github is a collection of correlated Github Actions defined in a single file to control sequence of events or tasks. A workflow is a single file; you may have multiple workflows for different things. An Action is a reusable packaging of a workflow. You create an action when you want to run the same workflow in multiple repositories.

In a workflow file, you define two things:

- What conditions trigger this? For instance, a merge onto master.
- What things should happen?

The "things that happen" are technically called jobs, and are then composed of steps.

Workflow is like a playlist where action is a song in that playlist.
