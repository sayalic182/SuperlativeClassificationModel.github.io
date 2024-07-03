from django.shortcuts import render,redirect,reverse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from app.auth import authentication
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import cache_control
from .csv_process import *
from .models import *
from django.http import JsonResponse
import os
from .image_process import *
from .video_process import *
from .pose_process import *
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from django.http import HttpResponse, FileResponse
import mimetypes
import zipfile
import json
# Create your views here.



def index(request):
    return render(request, 'index.html')

def register(request):
    if request.method == "POST":
        first_name = request.POST['fname']
        last_name = request.POST['lname']
        username = request.POST['email']
        password = request.POST['password']
        repassword = request.POST['repassword']
        # print(first_name, contact_no, ussername)
        verify = authentication(first_name, last_name, password, repassword )
        if verify == "success":
            user = User.objects.create_user(username = username, password = password)          #create_user
            user.email = username
            user.first_name = first_name
            user.last_name = last_name
            user.save()
            messages.success(request, "Your Account has been Created.")
            return redirect("log_in")
            
        else:
            messages.error(request, verify)
            return redirect("register")
            # return HttpResponse("This is Home page")
    return render(request, "register.html")

def log_in(request):
    if request.method == "POST":
        # return HttpResponse("This is Home page")  
        u_name = request.POST['username']
        password = request.POST['password']

        user = authenticate(username = u_name, password = password)
        if user is not None:
            login(request, user)
            messages.success(request, "Log In Successful...!")
            return redirect("projects")
        else:
            messages.error(request, "Invalid User...!")
            return redirect("log_in")
    return render(request, "log_in.html")


login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def dashboard(request):
    context = {
        'fname' : request.user.first_name
    }
    return render(request, 'dashboard.html', context)



@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def log_out(request):
    logout(request)
    messages.success(request, "Log out Successfuly...!")
    return redirect("/")


def get_image_paths(folder_path):
    """
    Get image paths within a folder.
    """
    image_paths = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            image_paths.append(file_path)
    return image_paths


@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def cards(request, model_name):
    context = {
        'fname': request.user.first_name
    }

    # Path to the objects folder within the model folder
    objects_folder = os.path.join(os.getcwd(), 'media', model_name, 'Objects')

    # Check if the objects folder exists
    if os.path.exists(objects_folder):
        subfolders = os.listdir(objects_folder)
        images_dict = {}

        # Iterate over subfolders
        for subfolder in subfolders:
            subfolder_path = os.path.join(objects_folder, subfolder)
            if os.path.isdir(subfolder_path):
                # Get image paths within the subfolder
                image_paths = get_image_paths(subfolder_path)
                # Store relative image paths in the dictionary
                images_dict[subfolder] = [os.path.relpath(image_path, 'media') for image_path in image_paths]

        # Pass the dictionary of image paths to the context
        context['images_dict'] = images_dict
    else:
        # If the objects folder does not exist, return an empty dictionary
        context['images_dict'] = {}

    if 'capture_image' in request.POST:
        class_name = request.POST.get('class_name')
        num_images = 200
        capture_images(model_name, class_name, num_images)
        messages.success(request, "Class Added Successfully!!!")
        return redirect(reverse("cards", kwargs={'model_name': model_name}))
    
    elif 'train' in request.POST:
        splits_folder = os.path.join(os.getcwd(), 'media', model_name, 'Splits')
        train_image(model_name, objects_folder,splits_folder)
        messages.success(request, "Your Model is Trained Successfully!!")
        return redirect(reverse("cards", kwargs={'model_name': model_name}))
    
    elif 'test' in request.POST:
        # Load the trained model
        model = load_model("models/" + model_name + ".h5")

        # Load class labels from JSON file
        with open( "jsons/" + model_name +  ".json", "r") as f:
            class_labels = json.load(f)
        
        test_image_model(model, class_labels)
        return redirect(reverse("cards", kwargs={'model_name': model_name}))
    
    elif 'download_model' in request.POST:
        # Paths to the files
        model_file_path = os.path.join(os.getcwd(), 'models', model_name + '.h5')
        json_file_path = os.path.join(os.getcwd(), 'jsons', model_name + '.json')
        code_file_path = os.path.join(os.getcwd(), 'codes', 'image_classification.py')

        # Prepare the response
        response = HttpResponse(content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="{0}.zip"'.format(model_name)

        # Create a zip file
        with zipfile.ZipFile(response, 'w') as zip_file:
            # Add the model file
            zip_file.write(model_file_path, arcname=os.path.basename(model_file_path))
            # Add the JSON file
            zip_file.write(json_file_path, arcname=os.path.basename(json_file_path))
            # Add the code file
            zip_file.write(code_file_path, arcname=os.path.basename(code_file_path))

        return response


    return render(request, 'cards.html', context)

@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def projects(request):
    context = {
        'fname': request.user.first_name
    }
    if request.method == "POST":
        if "image_train" in request.POST:
            print("Hello 1")
            project_name = request.POST['project_name']  # Get the project name from the request GET parameters
            model_name = request.POST['model_name']  # Get the model name from the request GET parameters
            
            if model_name:  # Check if model_name is not empty
                media_folder = os.path.join(os.getcwd(), 'media')  # Get the path to the media folder
                model_folder = os.path.join(media_folder, model_name)  # Create the path to the model folder
                
                # Check if the model folder already exists
                if os.path.exists(model_folder):
                    messages.info(request, "Project Already Exists")
                    return redirect("projects")  # Return a response indicating that the project already exists
                
                # If the model folder doesn't exist, create it
                os.makedirs(model_folder)
                messages.success(request, "Project Created Successfully")
                return redirect(reverse("cards", kwargs={'model_name': model_name}))  # Return a success response
        
        elif "video_train" in request.POST:
            print("Hello 2")
            project_name = request.POST['project_name']  # Get the project name from the request GET parameters
            model_name = request.POST['model_name']  # Get the model name from the request GET parameters
            
            if model_name:  # Check if model_name is not empty
                media_folder = os.path.join(os.getcwd(), 'static/videos')  # Get the path to the videos folder
                model_folder = os.path.join(media_folder, model_name)  # Create the path to the model folder
                
                # Check if the model folder already exists
                if os.path.exists(model_folder):
                    messages.info(request, "Project Already Exists")
                    return redirect("projects")  # Return a response indicating that the project already exists
                
                # If the model folder doesn't exist, create it
                os.makedirs(model_folder)
                messages.success(request, "Project Created Successfully")
                return redirect(reverse("videocards", kwargs={'model_name': model_name}))  # Return a success response
        
        elif "pose_train" in request.POST:
            project_name = request.POST['project_name']  # Get the project name from the request GET parameters
            model_name = request.POST['model_name']  # Get the model name from the request GET parameters
            
            if model_name:  # Check if model_name is not empty
                media_folder = os.path.join(os.getcwd(), 'pose')  # Get the path to the videos folder
                model_folder = os.path.join(media_folder, model_name)  # Create the path to the model folder
                
                # Check if the model folder already exists
                if os.path.exists(model_folder):
                    messages.info(request, "Project Already Exists")
                    return redirect("projects")  # Return a response indicating that the project already exists
                
                # If the model folder doesn't exist, create it
                os.makedirs(model_folder)
                messages.success(request, "Project Created Successfully")
                return redirect(reverse("posecards", kwargs={'model_name': model_name}))  # Return a success response
        
    return render(request, 'projects.html', context)

def videocards(request,model_name):
    context = {
        'fname': request.user.first_name
    }

    # Path to the objects folder within the model folder
    objects_folder = os.path.join(os.getcwd(), 'static/videos', model_name, 'Objects')
    print(objects_folder)
    # Check if the objects folder exists
    if os.path.exists(objects_folder):
        subfolders = os.listdir(objects_folder)
        video_dict = {}

        # Iterate over subfolders
        for subfolder in subfolders:
            subfolder_path = os.path.join(objects_folder, subfolder)
            if os.path.isdir(subfolder_path):
                # Get image paths within the subfolder
                image_paths = get_image_paths(subfolder_path)
                # Store relative image paths in the dictionary
                video_dict[subfolder] = [os.path.relpath(image_path, 'videos') for image_path in image_paths]
        print(video_dict)
        # Pass the dictionary of image paths to the context
        context['video_dict'] = video_dict
    else:
        # If the objects folder does not exist, return an empty dictionary
        context['video_dict'] = {}

    if 'capture_video' in request.POST:
        class_name = request.POST.get('class_name')
        num_videos = 20
        video_duration = 5
        capture_videos(model_name, class_name, num_videos, video_duration)
        messages.success(request, "Class Added Successfully!!!")
        return redirect(reverse("videocards", kwargs={'model_name': model_name}))
    
    elif 'train' in request.POST:
        splits_folder = os.path.join(os.getcwd(), 'static/videos', model_name, 'Splits')
        train_video(model_name, objects_folder,splits_folder)
        messages.success(request, "Your Model is Trained Successfully!!")
        return redirect(reverse("videocards", kwargs={'model_name': model_name}))
    
    elif 'test' in request.POST:
        # Load the trained model
        model = load_model("static/models/" + model_name + ".h5")

        # Load class labels from JSON file
        with open( "static/json/" + model_name +  ".json", "r") as f:
            class_labels = json.load(f)
        
        
        class_names = {v: k for k, v in class_labels.items()}
        test_video_model(model, class_names)
        return redirect(reverse("videocards", kwargs={'model_name': model_name}))
    
    elif 'download_model' in request.POST:
        # Paths to the files
        model_file_path = os.path.join(os.getcwd(), 'static/models', model_name + '.h5')
        json_file_path = os.path.join(os.getcwd(), 'static/json', model_name + '.json')
        code_file_path = os.path.join(os.getcwd(), 'codes', 'video_classification.py')

        # Prepare the response
        response = HttpResponse(content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="{0}.zip"'.format(model_name)

        # Create a zip file
        with zipfile.ZipFile(response, 'w') as zip_file:
            # Add the model file
            zip_file.write(model_file_path, arcname=os.path.basename(model_file_path))
            # Add the JSON file
            zip_file.write(json_file_path, arcname=os.path.basename(json_file_path))
            # Add the code file
            zip_file.write(code_file_path, arcname=os.path.basename(code_file_path))

        return response


    return render(request, 'videocards.html', context)


def posecards(request,model_name):
    context = {
        'fname': request.user.first_name
    }
    
    if 'capture_pose' in request.POST:
        class_name = request.POST.get('class_name')
        settle_duration = 15  # settlement time in seconds
        capture_duration = 10  # duration in seconds for capturing real poses
        filename = f'pose/{model_name}/{model_name}.csv'
        data = collect_data(class_name, capture_duration, settle_duration)
        save_data(data, filename)
        messages.success(request, "Class Added Successfully!!!")
        return redirect(reverse("posecards", kwargs={'model_name': model_name}))
    
    elif 'train' in request.POST:
        # Load collected data
        filename = f'pose/{model_name}/{model_name}.csv'
        pose_data = pd.read_csv(filename)
        knn, le = train_pose(pose_data)
        # Save the model and label encoder
        joblib.dump(knn, f'pose/{model_name}/knn_model.pkl')
        joblib.dump(le, f'pose/{model_name}/label_encoder.pkl')
        messages.success(request, "Your Model is Trained Successfully!!")
        return redirect(reverse("posecards", kwargs={'model_name': model_name}))
    
    elif 'test' in request.POST:
        knn = joblib.load(f'pose/{model_name}/knn_model.pkl')
        le = joblib.load(f'pose/{model_name}/label_encoder.pkl')

        test_pose(knn, le)
        return redirect(reverse("posecards", kwargs={'model_name': model_name}))
    
    elif 'download_model' in request.POST:
        # Paths to the files
        model_file_path = os.path.join(os.getcwd(), 'pose/', model_name + '/knn_model.pkl')
        json_file_path = os.path.join(os.getcwd(), 'pose/', model_name + '/label_encoder.pkl')
        code_file_path = os.path.join(os.getcwd(), 'codes', 'pose_classification.py')

        # Prepare the response
        response = HttpResponse(content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="{0}.zip"'.format(model_name)

        # Create a zip file
        with zipfile.ZipFile(response, 'w') as zip_file:
            # Add the model file
            zip_file.write(model_file_path, arcname=os.path.basename(model_file_path))
            # Add the JSON file
            zip_file.write(json_file_path, arcname=os.path.basename(json_file_path))
            # Add the code file
            zip_file.write(code_file_path, arcname=os.path.basename(code_file_path))

        return response


    return render(request, 'posecards.html', context)

@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def csv_train(request):
    context = {
        'fname' : request.user.first_name
    }
    if 'train' in request.POST : 
        model_name = request.POST['model_name']
        uploaded_file = request.FILES['uploaded_file']
        trained_model_name,feature_names,label_encoder_model_name = csv_trained(model_name, uploaded_file)
        save_model = CSV_Data.objects.create(trained_model_name = str(trained_model_name), feature_names = list(feature_names), label_encoder_model_name = str(label_encoder_model_name))
        save_model.save()
        messages.success(request, "Model Trained Successfuly...!")
        
        return redirect("csv_test")
    
    return render(request, 'csv_train.html', context)

@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def csv_test(request):
    train_model = CSV_Data.objects.last()
    feature_names = eval(train_model.feature_names)
    base_model_name = os.path.basename(train_model.trained_model_name)
    print(feature_names)
    file1_path = f'{train_model.trained_model_name}_1.txt'
    with open(file1_path, 'w') as file:
        for item in feature_names:
            file.write(f"{item}\n")
    context = {
        'fname' : request.user.first_name,
        'feature_names' : feature_names
    }

    if request.method == 'POST':
        if 'test' in request.POST :
            # Access input values from the form
            feature_values = {}
            for feature, value in request.POST.items():
                if feature != 'csrfmiddlewaretoken' and value:  # Exclude CSRF token and check if value is not empty
                    feature_values[feature] = float(value)
            
            pred = test_model(train_model.trained_model_name, feature_values, train_model.label_encoder_model_name)
            context['pred'] = pred[0]
            messages.success(request, f"Prediction : {pred[0]}")
        
        elif 'download_model' in request.POST:
            # Paths to the files
            model_file_path = os.path.join(os.getcwd(), 'models/', base_model_name)
            label_encoder_file_path = os.path.join(os.getcwd(), 'models/', 'label_encoder_' + base_model_name)
            code_file_path = os.path.join(os.getcwd(), 'codes/', 'csv_classification' + '.py')
            json_file_path = os.path.join(os.getcwd(),'models/', base_model_name + '_1.txt')

            # Prepare the response
            response = HttpResponse(content_type='application/zip')
            response['Content-Disposition'] = 'attachment; filename="{0}.zip"'.format(train_model.trained_model_name)

            # Create a zip file
            with zipfile.ZipFile(response, 'w') as zip_file:
                # Add the model file
                zip_file.write(model_file_path, arcname=os.path.basename(model_file_path))
                # Add the model file
                zip_file.write(label_encoder_file_path, arcname=os.path.basename(label_encoder_file_path))
                # Add the JSON file
                zip_file.write(json_file_path, arcname=os.path.basename(json_file_path))
                # Add the code file
                zip_file.write(code_file_path, arcname=os.path.basename(code_file_path))

            return response

    return render(request, 'csv_test.html', context)


def train_model(request):
    if request.method == 'POST':
        data = request.POST.dict()
        print(data)
        # model_name = data['model']
        # classes = data['classes']

        # # Create directory for the model
        # model_directory = os.path.join(settings.MEDIA_ROOT, model_name)
        # os.makedirs(model_directory, exist_ok=True)

        # # Create subdirectories for each class and save images
        # for class_data in classes:
        #     class_name = class_data['name']
        #     class_directory = os.path.join(model_directory, class_name)
        #     os.makedirs(class_directory, exist_ok=True)

        #     # Save images to class directory
        #     for index, image_url in enumerate(class_data['images']):
        #         image_content = requests.get(image_url).content  # If image URL is provided
        #         with open(os.path.join(class_directory, f'image_{index}.jpg'), 'wb') as f:
        #             f.write(image_content)

        return JsonResponse({'message': 'Model trained successfully'}, status=200)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)