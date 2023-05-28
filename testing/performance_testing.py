from locust import HttpUser, between, task


# Create a Locust User class that represents the behavior of the virtual users in your test
class MyUser(HttpUser):
    # By specifying between(3, 3), it means that each virtual user will wait for
    # 3 seconds before sending the next request.
    # This is in consistent with our assumption that each user will make an api request every 3 seconds.
    wait_time = between(3, 3)

    @task
    def make_api_request(self):
        # since the ml model needs an image for performing predictions,
        # we need to send an image, which is from our dataset, in the post request.

        file_path = "../dataset/000007.png"
        # Open the image file
        with open(file_path, "rb") as file:
            # Create a dictionary with the file data
            files = {"frame": file}

            # Send the POST request with the image file
            self.client.post("/predict", files=files)
