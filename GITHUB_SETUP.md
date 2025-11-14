# GitHub Repository Setup Guide

Quick guide to upload this project to GitHub for Claude Code Web to continue.

---

## Step 1: Create GitHub Repository

### Option A: Via GitHub Website
1. Go to https://github.com
2. Click "+" → "New repository"
3. Repository name: `bivector-framework`
4. Description: "Exploring universal exp(-Λ²) patterns via bivector algebra"
5. Choose Public (for collaboration) or Private
6. **DO NOT** initialize with README (we have one)
7. Click "Create repository"

### Option B: Via GitHub CLI
```bash
gh repo create bivector-framework --public --description "Bivector framework exploration"
```

---

## Step 2: Initialize Local Git Repository

```bash
# Navigate to project directory
cd C:\v2_files\hierarchy_test

# Initialize git (if not already done)
git init

# Add all files
git add .

# Check what will be committed
git status

# Make initial commit
git commit -m "Initial commit: Bivector framework with BCH validation (R²=1.000)"
```

---

## Step 3: Connect to GitHub

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/bivector-framework.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Step 4: Verify Upload

Go to: `https://github.com/YOUR_USERNAME/bivector-framework`

You should see:
- ✅ README.md displayed on front page
- ✅ All .py files
- ✅ Documentation files (.md)
- ✅ SPRINT.md with sprint plan

---

## Step 5: Set Up for Claude Code Web

### Create New Issue with Sprint Plan

1. Go to your repository
2. Click "Issues" tab
3. Click "New issue"
4. Title: "Sprint: Bivector Pattern Hunter - 5 Day Exploration"
5. Copy the SPRINT.md content into issue description
6. Add labels: `enhancement`, `good first issue`, `help wanted`
7. Create issue

### Tag the Issue for Claude Code

Add this to top of issue:
```
@claude-code Please execute this 5-day sprint to explore unexplored bivector combinations.

Key Context:
- BCH materials validation: R² = 1.000 (SOLID)
- Literal 5D interpretation: FALSIFIED (don't revisit)
- Goal: Find new exp(-Λ²) correlations systematically
- Deliverables: 5 new .py files + comprehensive documentation

Start with Day 1 (atomic physics) and proceed sequentially.
```

---

## Step 6: Optional - Add Useful Files

### Create placeholder directories
```bash
mkdir results
mkdir figures
mkdir data

# Create .gitkeep files to preserve structure
touch results/.gitkeep
touch figures/.gitkeep
touch data/.gitkeep

git add results/.gitkeep figures/.gitkeep data/.gitkeep
git commit -m "Add directory structure for results"
git push
```

### Add requirements.txt
```bash
# Create requirements.txt
cat > requirements.txt << EOF
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
pandas>=1.3.0
scikit-learn>=0.24.0
# Optional for ML (Day 5)
# pysr>=0.11.0
# gplearn>=0.4.0
EOF

git add requirements.txt
git commit -m "Add Python dependencies"
git push
```

---

## Step 7: Clone on Another Machine (Optional)

To work on different computer or share with collaborators:

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/bivector-framework.git
cd bivector-framework

# Install dependencies
pip install -r requirements.txt

# Verify setup
python bivector_systematic_search.py

# You're ready to go!
```

---

## Step 8: Using Claude Code Web

### Via Claude.ai/code:

1. Go to https://claude.ai/code
2. Start new project
3. Connect to GitHub repository:
   - Click "Connect to GitHub"
   - Select `bivector-framework` repository
   - Grant permissions

4. Reference the sprint:
   ```
   I'd like to execute the SPRINT.md plan systematically.
   Start with Day 1: Atomic Physics Bivector Survey.

   Please:
   1. Read COMPREHENSIVE_SUMMARY.md for full context
   2. Review bivector_systematic_search.py for code patterns
   3. Create atomic_bivector_survey.py following the plan
   4. Test against NIST atomic spectra data
   5. Document results in day1_results.json
   ```

5. Claude Code will:
   - Read existing files for context
   - Create new files following templates
   - Run tests and calculations
   - Commit results back to repository
   - Generate visualizations
   - Document findings

---

## Git Workflow During Sprint

### Daily Commits
```bash
# After each day's work
git add .
git commit -m "Day X complete: [brief summary]"
git push

# Example:
git commit -m "Day 1 complete: Atomic bivector survey, 12 pairs tested, 3 correlations found"
```

### Branch Strategy (Optional)
```bash
# Create feature branch for sprint
git checkout -b sprint-bivector-hunter
# Work on sprint...
git push -u origin sprint-bivector-hunter

# When sprint complete, create PR
# Merge to main after review
```

---

## Collaboration Tips

### If Working with Others
```bash
# Pull latest changes before starting work
git pull origin main

# Push frequently (don't lose work!)
git push

# If conflicts occur
git pull --rebase origin main
# Resolve conflicts
git push
```

### Code Review
- Create Pull Requests for major changes
- Use Issues for questions/discussion
- Tag commits with issue numbers: `git commit -m "Fixes #1: Complete Day 1 atomic survey"`

---

## Troubleshooting

### Issue: Files too large
```bash
# GitHub has 100MB file limit
# If you have large data files:

# Add to .gitignore
echo "data/*.hdf5" >> .gitignore
echo "results/*.csv" >> .gitignore

# Remove from git (keep local copy)
git rm --cached large_file.csv
git commit -m "Remove large data file from tracking"
```

### Issue: Authentication failed
```bash
# Use personal access token instead of password
# Create token at: https://github.com/settings/tokens

# Use token as password when prompted
# Or set up SSH keys (more secure):
ssh-keygen -t ed25519 -C "your_email@example.com"
# Add public key to GitHub
```

### Issue: Merge conflicts
```bash
# Pull changes
git pull origin main

# If conflicts:
# 1. Open conflicted files
# 2. Look for <<<<<<< HEAD markers
# 3. Choose which version to keep
# 4. Remove conflict markers
# 5. git add [resolved_files]
# 6. git commit -m "Resolve merge conflict"
```

---

## Quick Reference Commands

```bash
# Status
git status                    # What's changed?
git log --oneline            # Commit history

# Staging
git add file.py              # Stage specific file
git add .                    # Stage all changes
git reset file.py            # Unstage file

# Committing
git commit -m "message"      # Commit with message
git commit --amend           # Fix last commit

# Pushing/Pulling
git push                     # Send to GitHub
git pull                     # Get from GitHub
git fetch                    # Check for updates

# Branching
git branch                   # List branches
git branch feature-name      # Create branch
git checkout feature-name    # Switch branch
git merge feature-name       # Merge branch

# Undoing
git checkout -- file.py      # Discard changes
git reset --hard HEAD        # Reset everything (dangerous!)
git revert commit-hash       # Undo specific commit
```

---

## Repository Best Practices

### 1. Commit Messages
Good:
```
✅ "Day 1 complete: Tested 12 atomic bivector pairs, found 3 correlations"
✅ "Fix: Correct Λ calculation for time-dependent bivectors"
✅ "Add: ML pattern discovery (Day 5)"
```

Bad:
```
❌ "update"
❌ "fix bug"
❌ "asdfasdf"
```

### 2. Commit Frequency
- Commit after each logical unit of work
- Not too frequent (every line) or too rare (once per week)
- Good rule: If you'd be upset to lose this work, commit it!

### 3. Documentation
- Update README.md when adding features
- Document negative results (important!)
- Include code comments for complex calculations
- Keep SPRINT.md updated with progress

### 4. Code Quality
- Test before committing (run the code!)
- Check for syntax errors
- Include docstrings
- Follow consistent style

---

## After Sprint: Sharing Results

### Create Release
```bash
# After sprint complete
git tag -a v1.0-sprint-results -m "Bivector Pattern Hunter Sprint Results"
git push origin v1.0-sprint-results
```

### Write Summary
Create `SPRINT_RESULTS.md`:
```markdown
# Sprint Results: Bivector Pattern Hunter

## Summary
- Bivector pairs tested: 24
- New correlations found: 5
- Best R²: 0.943 (condensed matter)
- Surprise finding: [description]

## Key Files
- `atomic_bivector_survey.py`: [results]
- `pattern_synthesis_ml.py`: [discoveries]

## Next Steps
- [Follow-up investigations]
- [New hypotheses]
```

Commit and push!

---

## Resources

### Git Learning
- Git Tutorial: https://git-scm.com/docs/gittutorial
- GitHub Guides: https://guides.github.com/
- Interactive Tutorial: https://learngitbranching.js.org/

### GitHub Features
- Issues: Track tasks and bugs
- Projects: Kanban-style task boards
- Actions: Automate testing (advanced)
- Pages: Host documentation website

### Claude Code
- Docs: https://docs.anthropic.com/claude/docs/claude-code
- Examples: https://github.com/anthropics/claude-code-examples

---

## Final Checklist

Before sharing with Claude Code Web:

- [ ] Repository created on GitHub
- [ ] All files pushed (`git push`)
- [ ] README.md is clear and informative
- [ ] SPRINT.md is in repository
- [ ] .gitignore configured properly
- [ ] requirements.txt added
- [ ] Issue created for sprint
- [ ] Repository is public (or collaborators invited)
- [ ] Tested cloning on fresh machine

---

**You're all set!** 🚀

Share the repository URL with Claude Code and let the exploration begin!

**Repository URL format**: `https://github.com/YOUR_USERNAME/bivector-framework`

Good luck with the sprint! 🎯
