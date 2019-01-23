from whale.forms import ProfileForm 
from whale.models import Profile 
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse


@csrf_exempt 
def index(request):
    context = {}
    form = ProfileForm
    context['form'] = form 
    return render(request, 'index.html', context)

@csrf_exempt 
def save_profile(request):
    if request.method == "POST":
        print(request.POST)
        print(request.FILES)
        form = ProfileForm(request.POST, request.FILES)
        print(form)
        if form.is_valid():
            profile = Profile()
            profile.picture = form.cleaned_data["picture"]
            profile.save()
            return HttpResponse('123')
        else:
            form = ProfileForm()
            return HttpResponse('321')







