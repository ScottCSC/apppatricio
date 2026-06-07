
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get("https://www.google.com")
search_box = driver.find_element(By.NAME, "q")
search_box.send_keys("automatización con Selenium")
search_box.submit()
driver.quit()
