# Generated by Django 3.2.5 on 2021-07-30 07:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0003_auto_20210728_0843'),
    ]

    operations = [
        migrations.CreateModel(
            name='PrediksiDataModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pesan', models.TextField()),
                ('label', models.CharField(max_length=255)),
            ],
        ),
    ]
