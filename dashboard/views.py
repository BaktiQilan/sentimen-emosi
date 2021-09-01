from django.contrib import messages
from django.db.utils import OperationalError
from .models import DatasetModel, NewDataModel, PrediksiDataModel
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import render
import pandas as pd
import os
import csv
from django.views.generic import TemplateView, ListView
from .predict import prediksi, kamus_slang, prediksi_list, preprocess, retrain, coba, wordCloudMarah, wordCloudPercaya, wordCloudSedih, wordCloudSenang, wordCloudTakut
from django.utils.translation import gettext_lazy as _
from django.core.files.storage import FileSystemStorage
from whatstk import WhatsAppChat
from django.conf import settings
from django.db import connections
import tweepy
import matplotlib.pyplot as plt

    

class DashboardPageView(LoginRequiredMixin, ListView):
    template_name = 'dashboard/dashboard.html'
    #model = DatasetModel
    context_object_name = 'dataset'
    queryset = DatasetModel.objects.all()
    paginate_by = 5

    def get_context_data(self,**kwargs):
        context = super(DashboardPageView, self).get_context_data(**kwargs)

        # untuk memasukan dataset ke database
        # base_dir = settings.BASE_DIR  
        # file_path = os.path.join(base_dir, 'content/media/dataset/dataset.csv')
        # df = pd.read_csv(file_path, sep=',')
        # for row in df.itertuples():
        #     row = DatasetModel.objects.create(pesan=row.pesan,label=row.label)

        data_df_pesan =  list(DatasetModel.objects.all().values('pesan'))
        data_df_label =  list(DatasetModel.objects.all().values('label'))
        evaluasi = coba(data_df_pesan, data_df_label)

       
        context['model'] = evaluasi[0]
        context['akurasi'] = evaluasi[1]
        context['jumlah'] =  DatasetModel.objects.all().count()
        return context




class ModelingPageView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard/modeling.html'

    def post(self, request):

        if 'get' in request.POST:
            try:
                cursor = connections['api_iteung'].cursor()
                cursor.execute("SELECT message FROM data ORDER BY timestamps DESC LIMIT 10")
                # SELECT message FROM data WHERE message REGEXP '^[^ ]+[ ]+[^ ]+[ ]+[^ ]$' ORDER BY timestamps DESC LIMIT 10
                #   SELECT MIN(`message`) as message FROM data  
                #   WHERE `message` REGEXP '^[^ ]+[ ]+[^ ]+[ ]+[^ ]+$' 
                #   GROUP BY `message` 
                #   ORDER BY `timestamps` DESC LIMIT 10
                def dictfetchall(cursor):
                # "Return all rows from a cursor as a dict"
                    columns = [col[0] for col in cursor.description]
                    return [
                        dict(zip(columns, row))
                        for row in cursor.fetchall()
                    ]


                rwa = dictfetchall(cursor)
                df = pd.DataFrame(rwa)
                df.columns = ['pesan']
                check = NewDataModel.objects.filter(id=None)
                if check == True:
                    for index, row in df.iterrows():
                        model = NewDataModel()
                        model.pesan = row['pesan']
                        model.save()
                else:
                    NewDataModel.objects.all().delete()
                    for index, row in df.iterrows():
                        model = NewDataModel()
                        model.pesan = row['pesan']
                        model.save()
                show_data_2 = NewDataModel.objects.all()
                context = {'show_data': show_data_2}
                return render(request, 'dashboard/modeling.html', context)  
            except OperationalError:
                messages.error(request, _('Database Iteung tidak terhubung'))

        
        elif 'scrap' in request.POST:
            consumer_key = settings.CONSUMER_KEY
            consumer_secret = settings.CONSUMER_SECRET
            access_token = settings.ACCESS_TOKEN
            access_token_secret = settings.ACCESS_TOKEN_SECRET

            keyword = request.POST.get("keyword")

            auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
            auth.set_access_token(access_token,access_token_secret)
            api = tweepy.API(auth)
            try:
                tweets = api.search(keyword, lang='in', count='5')
                pd.set_option('display.max_colwidth', None)
                data = pd.DataFrame()
                data['pesan'] = [tweets.text for tweets in tweets]

                check = NewDataModel.objects.filter(id=None)
                if check == True:
                    for index, row in data.iterrows():
                        model = NewDataModel()
                        model.pesan = row['pesan']
                        model.save()
                else:
                    NewDataModel.objects.all().delete()
                    for index, row in data.iterrows():
                        model = NewDataModel()
                        model.pesan = row['pesan']
                        model.save()
            except tweepy.TweepError:
                messages.error(request, _('Rate limit exceeded'))

            show_data_2 = NewDataModel.objects.all()
            return render(request,'dashboard/modeling.html', {'show_data':show_data_2})

        elif 'watxt' in request.FILES:
            name = 'exportwa.txt'
            fss = FileSystemStorage()
            file = request.FILES['watxt']
            check = fss.exists(name)
            

            if check == True:
                os.remove(os.path.join(settings.MEDIA_ROOT, 'exportwa.txt'))
                fss.save('exportwa.txt', file)
            else:
                fss.save('exportwa.txt', file)
            
            try:
                filewa = os.path.join(settings.MEDIA_ROOT, 'exportwa.txt')
                pd.set_option('display.max_colwidth', None)
                chat = WhatsAppChat.from_source(filepath=filewa)
                
                
                check2 = NewDataModel.objects.filter(id=None)
                if check2 == True:
                    for index, row in chat.df.iterrows():
                        model = NewDataModel()
                        model.pesan = row['message']
                        model.save()
                else:
                    NewDataModel.objects.all().delete()
                    for index, row in chat.df.iterrows():
                        model = NewDataModel()
                        model.pesan = row['message']
                        model.save()
                data_wa = NewDataModel.objects.all()
                return render(request,'dashboard/modeling.html', {'show_data':data_wa})
            except RuntimeError:
                messages.error(request, _('File format does not match, please press the button how to get file !'))


        elif 'slang' in request.POST:
            data_list = list(NewDataModel.objects.all().values('pesan'))
            pd.set_option('display.max_colwidth', None)
            data = pd.DataFrame(data_list)
            data['pesan'] = data['pesan'].apply(kamus_slang)
            check = NewDataModel.objects.filter(id=None)
            if check == True:
                for index, row in data.iterrows():
                    model = NewDataModel()
                    model.pesan = row['pesan']
                    model.save()
            else:
                NewDataModel.objects.all().delete()
                for index, row in data.iterrows():
                    model = NewDataModel()
                    model.pesan = row['pesan']
                    model.save()
            slang_data = NewDataModel.objects.all()
            return render(request,'dashboard/modeling.html', {'slang':slang_data})

        elif 'preprocess' in request.POST:
            data_list = list(NewDataModel.objects.all().values('pesan'))
            pd.set_option('display.max_colwidth', None)
            data = pd.DataFrame(data_list)
            data['pesan'] = data['pesan'].apply(preprocess)
            check = NewDataModel.objects.filter(id=None)
            if check == True:
                for index, row in data.iterrows():
                    model = NewDataModel()
                    model.pesan = row['pesan']
                    model.save()
            else:
                NewDataModel.objects.all().delete()
                for index, row in data.iterrows():
                    model= NewDataModel()
                    model.pesan = row['pesan']
                    model.save()
            preprocessing = NewDataModel.objects.all()
            return render(request, 'dashboard/modeling.html', {'prepro': preprocessing})

        elif 'prediksi' in request.POST:
            data_list = list(NewDataModel.objects.all().values('pesan'))
            pd.set_option('display.max_colwidth', None)
            data = pd.DataFrame(data_list)
            jadi = data['pesan'].tolist() 
            pred = prediksi_list(jadi)
            df = pd.DataFrame({'pesan':jadi,'label':pred})
            check = PrediksiDataModel.objects.filter(id=None)
            if check == True:
                for row in df.itertuples():
                    row = PrediksiDataModel.objects.create(pesan=row.pesan,label=row.label)
            else:
                PrediksiDataModel.objects.all().delete()
                for row in df.itertuples():
                    row = PrediksiDataModel.objects.create(pesan=row.pesan,label=row.label)
                
            prediksi = PrediksiDataModel.objects.all()
            return render(request, 'dashboard/modeling.html', {'prediksi':prediksi})

        elif 'hasil' in request.POST:
            cek = request.POST.getlist('label') 
            data_list = list(PrediksiDataModel.objects.all().values('pesan'))
            data = pd.DataFrame(data_list)
            jadi = data['pesan'].tolist()
            df = pd.DataFrame({'pesan':jadi,'label':cek})
            for index, row in df.iterrows():
                    model= DatasetModel()
                    model.pesan = row['pesan']
                    model.label = row['label']
                    model.save()

            
            pie = list(DatasetModel.objects.all().values('label'))
            pie_df = pd.DataFrame(pie)
            sentiments = ['marah', 'sedih', 'takut', 'senang', 'percaya'] 
            slices = [(pie_df['label'] == 'marah').sum(), 
                    (pie_df['label'] == 'sedih').sum(), (pie_df['label'] == 'takut').sum(),
                    (pie_df['label'] == 'senang').sum(), (pie_df['label'] == 'percaya').sum()] 
            colors = ['darkred', 'darkmagenta', 'darkcyan', 'goldenrod', 'darkblue'] 
            
            fig, ax = plt.subplots()
            fig.patch.set_facecolor('#25282c')
            patches, texts, pcts = ax.pie(slices, labels=sentiments, colors=colors, startangle=90, radius = 1.5, autopct = '%1.2f%%')
            plt.setp(pcts, color='whitesmoke')
            legend = plt.legend(sentiments, facecolor='#25282c', bbox_to_anchor=(1,0.5), loc="center right", bbox_transform=plt.gcf().transFigure)
            plt.setp(legend.get_texts(), color='whitesmoke')

            base_dir = settings.BASE_DIR
            file_path = os.path.join(base_dir, 'content/assets/img/pie_chart.png')
            plt.savefig(file_path, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close()
            
            data_df_pesan =  list(DatasetModel.objects.all().values('pesan'))
            data_df_label =  list(DatasetModel.objects.all().values('label'))
            retrain(data_df_pesan, data_df_label)
            wordCloudMarah(data_df_pesan, data_df_label)
            wordCloudSenang(data_df_pesan, data_df_label)
            wordCloudSedih(data_df_pesan, data_df_label)
            wordCloudPercaya(data_df_pesan, data_df_label)
            wordCloudTakut(data_df_pesan, data_df_label)
            
            return render(request,'dashboard/modeling.html')                  
                

        return render(request,'dashboard/modeling.html')

class GuidePageView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard/guide.html'


class PredictPageView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard/predict.html'

    def post(self, request, *args, **kwargs):
        if request.method == 'POST':
            kalimat = request.POST.get("kalimat")
            pred = prediksi(kalimat)
            if pred is not None:
                return render(request, 'dashboard/predict.html', {'prediction':pred, 'kaaa':kalimat})
        return render(request, 'dashboard/predict.html')

class SlangPageView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard/slang_dictionary.html'

    def get(self, request):
        base_dir = settings.BASE_DIR  
        file_path = os.path.join(base_dir, 'content/media/dataset/slang_word.txt')   #full path to text.
        # data_file = open(file_path , 'r')       
        # data = data_file.read().split('\n')
        # context = {'dictionary': data}
        item = []

        with open(file_path, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='=', quotechar='|')
            for row in spamreader:
                tmp = {}
                tmp['word']    = row[0]
                tmp['transform']  = row[1]
                item.append(tmp)
                
        context = {'dictionary': item}
        return render(request, 'dashboard/slang_dictionary.html',context)
