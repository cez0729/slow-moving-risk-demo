# GitHub 发布步骤

## 1. 进入独立仓库

```powershell
cd C:\path\to\slow-moving-risk-demo
```

确认当前目录是新建的 `slow-moving-risk-demo`，不要在公司项目目录执行初始化。

## 2. 重新生成并验证

```powershell
python -m pip install -e ".[dev]"
python scripts/generate_demo_data.py
python -m ruff check src scripts tests
python -m pytest -q
python scripts/train.py --config configs/demo.yaml
python scripts/predict.py --config configs/demo.yaml
```

## 3. 发布前检查

```powershell
rg -n -i "1768285562|公司名称|C:\\Users\\|password|api[_-]?key|secret|token" .
git status --short --ignored
```

命令输出中不应出现真实公司名称、内部绝对路径、凭据或公司数据文件。`data/demo.csv`、`artifacts/` 和模型文件应保持 ignored。

## 4. 初始化 Git 仓库

```powershell
git init
git add README.md LICENSE pyproject.toml .gitignore configs src scripts tests data/README.md docs .github
git status --short
git commit -m "Create sanitized first-event slow-moving risk demo"
```

提交前人工检查 `git status`，确认没有 `*.xlsx`、`*.csv`（除非你明确希望提交一个经过审计的小型公开数据集）、`*.joblib`、内部报告或公司路径。

## 5. 创建 GitHub 远程仓库并推送

在 GitHub 网页创建空仓库 `slow-moving-risk-demo`，不要自动生成 README、License 或 `.gitignore`，然后执行：

```powershell
git branch -M main
git remote add origin https://github.com/<your-account>/slow-moving-risk-demo.git
git push -u origin main
```

将 `<your-account>` 替换为自己的 GitHub 用户名。不要把 token 写入源码、配置或远程 URL；使用 Git Credential Manager 或 GitHub CLI 登录。

## 6. GitHub 页面检查

确认 README 首屏说明 synthetic data，Actions 的 CI 运行通过，仓库中没有公司名称、真实指标、真实商品/供应商标识或模型文件。
