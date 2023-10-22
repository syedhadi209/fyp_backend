from django.urls import path
from .views import GetTraits,FetchTweets

urlpatterns = [
    path('get-traits/', GetTraits.as_view()),
    path('fetch-tweets/', FetchTweets.as_view())
]