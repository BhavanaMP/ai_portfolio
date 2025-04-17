import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { LoginComponent } from './login/login.component';
import { SearchUIComponent } from './search-ui/search-ui.component';
import { ResultsUiComponent } from './results-ui/results-ui.component';
import { ServletService } from './Service/servlet.service';
import { MatDialogModule } from '@angular/material/dialog';
import { MatTableModule } from '@angular/material/table';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatChipsModule } from '@angular/material/chips';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import {BrowserAnimationsModule} from '@angular/platform-browser/animations';
import { MatAutocompleteModule} from '@angular/material/autocomplete';
import { NgMatSearchBarModule } from 'ng-mat-search-bar';
import {MatToolbarModule} from '@angular/material/toolbar';
import {MatCardModule} from '@angular/material/card';
import { RouterModule } from '@angular/router';
//import { Ng4LoadingSpinnerModule } from 'ng4-loading-spinner';
import {MatGridList, MatGridListModule} from '@angular/material/grid-list';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { NgxSpinnerModule } from "ngx-spinner";


@NgModule({
  declarations: [
    AppComponent,
    LoginComponent,
    SearchUIComponent,
    ResultsUiComponent,
    // ServletService
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    FormsModule,
    ReactiveFormsModule,
    MatIconModule, MatInputModule,
    MatChipsModule,NgxSpinnerModule,
    MatFormFieldModule,BrowserAnimationsModule,MatAutocompleteModule,
    NgMatSearchBarModule,MatToolbarModule,MatCardModule,RouterModule,MatGridListModule, NgbModule
  ],
  providers: [//ServletService
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
