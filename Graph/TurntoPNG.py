from selenium import webdriver

driver_path = 'chromedriver.exe'

driver = webdriver.Chrome(driver_path)

driver.get('file:///path/to/your/html/file.html')

driver.implicitly_wait(10)

driver.save_screenshot('screenshot.png')

driver.quit()