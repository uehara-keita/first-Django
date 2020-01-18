from django.shortcuts import render, redirect

# Create your views here.
# ウィンドウに表示する要素を指定する
from django.http import HttpResponse
from django.template import loader
from .forms import PhotoForm
from .models import Photo

def index(request):
    template = loader.get_template("classai/index.html")
    context={"form":PhotoForm()}
    return HttpResponse(template.render(context,request))

def predict(request):
    # エラー処理
    if not request.method=="POST":
        return redirect("classai:index")
    form=PhotoForm(request.POST,request.FILES)
    if not form.is_valid():
        raise ValueError("Formが不正です")
    # メイン処理
    photo=Photo(image=form.cleaned_data["image"])
    predicted,percentage=photo.predict()

    template=loader.get_template("classai/result.html")

    context={
        "photo_name":photo.image.name,
        "photo_data":photo.image_src(),
        "predicted":predicted,
        "percentage":percentage,
    }
    return HttpResponse(template.render(context, request))
