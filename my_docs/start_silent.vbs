Set objShell = CreateObject("Wscript.Shell")
' 切换到项目目录
objShell.CurrentDirectory = "C:\Users\17947\Desktop\my_docs"
' 启动 Ollama（如果已开机自启可省略）
objShell.Run "cmd /c ollama serve > nul 2>&1", 0, False
' 等待 2 秒
WScript.Sleep 2000
' 启动 Streamlit（完全不显示窗口）
objShell.Run "cmd /c streamlit run app.py --server.headless true", 0, False
' 自动打开浏览器
objShell.Run "cmd /c start http://localhost:8501", 0, False