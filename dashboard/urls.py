from django.urls import path
from .views import DashboardPageView, GuidePageView, ModelingPageView, PredictPageView


app_name = 'dashboard'

urlpatterns = [
    path('', DashboardPageView.as_view(), name='dashboard'),
    path('guide/', GuidePageView.as_view(), name='guide'),
    path('modeling/', ModelingPageView.as_view(), name='modeling'),
    path('prediksi/', PredictPageView.as_view(), name='predict')
    #path('slang/', SlangPageView.as_view(), name='slang')

]


