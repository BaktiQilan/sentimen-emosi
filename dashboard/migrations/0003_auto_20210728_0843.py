# Generated by Django 3.2.5 on 2021-07-28 01:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0002_datasetmodel'),
    ]

    operations = [
        migrations.CreateModel(
            name='NewDataModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pesan', models.TextField()),
            ],
        ),
        migrations.AlterModelOptions(
            name='datasetmodel',
            options={'ordering': ['pesan']},
        ),
    ]
