# Start backend server
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; ./venv/Scripts/Activate; python main.py"

# Start frontend server
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"

Write-Host "Servers started!"
Write-Host "Frontend: http://localhost:3000"
Write-Host "Backend: http://localhost:8000" 